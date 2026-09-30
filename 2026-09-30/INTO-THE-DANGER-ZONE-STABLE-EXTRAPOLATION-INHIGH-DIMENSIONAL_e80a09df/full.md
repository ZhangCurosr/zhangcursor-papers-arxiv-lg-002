# INTO THE DANGER ZONE: STABLE EXTRAPOLATION INHIGH-DIMENSIONAL FUNCTION AND OPERATOR LEARNING

BEN ADCOCK<sup>†</sup>, SIMONE BRUGIAPAGLIA<sup>‡</sup>, AND XUEMENG WANG<sup>†</sup>

Abstract. Out-of-distribution (OOD) generalization is a central challenge in scientific machine learning. We study regression problems in which the test distribution difers from the training distribution and ask: under what assumptions on the target function or operator is stable extrapolation possible, and how far beyond the training domain can one extrapolate? Existing theory controls the test error through additive penalties measuring the discrepancy between the training and test distributions. Such guarantees show robustness to small distribution shifts, but can very pessimistic in comparison to OOD performance observed empirically. We identify classes of holomorphic functions and operators for which the OOD generalization error converges at algebraic rates even in the presence of large distribution shifts. This phenomenon stems from the increasing smoothness of higher-index coordinates, leading to what we term a ‘blessing of high dimensionality’. For learning with either polynomials, deep neural networks or deep neural operators, we derive explicit rates for arbitrary test measures supported on suitable domains and quantify how the admissible domain depends on the underlying regularity of the function or operator. Our extrapolation guarantees are independent of the test distribution, depending only on its support. We also present a series of numerical experiments across a range of functions and operators that support the main theoretical findings.

Key words. out-of-distribution generalization, stable extrapolation, high-dimensional approximation, operator learning, holomorphic functions and operators

MSC codes. 41A63, 41A10, 46E50, 68Q32, 68T07

1. Introduction. The ability of models to extrapolate beyond their training domain is a central challenge in modern Machine Learning (ML) and, in particular, Scientific ML (SciML). Understanding and enhancing the out-of-distribution (OOD) performance of ML models – namely, how well a model generalizes to examples drawn from a test distribution that difers from the training distribution – is a substantial area of research. Yet, as we discuss in §1.1, existing theory of OOD generalization in ML largely addresses the case of small distributional shifts, where the training and test distributions are close in some metric. Such guarantees are very general, placing minimal assumptions on the model or target object, but say little about larger shifts. However, a growing body of work in the SciML literature (described further in §1.4) has shown that trained Deep Neural Network (DNN) and Deep Neural Operator (DNO) models can often extrapolate: they generalize well much further from their training distribution than such pessimistic guarantees imply.

This raises a key question: under what assumptions on the target function or operator is stable extrapolation possible and how far can one extrapolate? We address this question for classes of holomorphic functions and operators using polynomials, DNNs and DNOs. Our results establish explicit algebraic convergence rates for the OOD generalization error in the L<sup>∞</sup>-norm and quantify how the admissible extrapolation domain depends on the underlying regularity Our results are distribution-agnostic, as they depend on the support of the test distribution only. They also reveal a certain blessing of high dimensionality: increasing smoothness in higher-index coordinates permits extrapolation in increasingly large domains and, in appropriate regimes, preserves the underlying algebraic convergence order. Our results are complemented with numerical experiments illustrating extrapolation in both function and operator learning and supporting our main conclusions.

This work combines classical numerical analysis of polynomial extrapolation, modern approximation theory in high and infinite dimensions and the recent advances in DNN/DNO approximation and generalization theory. We provide a coherent and comprehensive answer to this question of OOD generalization in the setting holomorphic regularity – a common assumption in practice, especially in SciML – thereby bridging a key gap between existing distributional shift theory in ML and practical performance of SciML models.

1.1. OOD generalization in ML. Consider a standard regression problem in ML where a model $\hat { f }$ is trained to approximate an unknown $f : \mathbb { R } ^ { d }  \mathbb { R }$ . Let $\varrho$ and $\mu$ be two probability measures on $\mathbb { R } ^ { d }$ , the in-distribution and out-of-distribution measures, respectively. Then typical OOD guarantees in ML (see §1.4 for references) take the form

$$
\begin{array} { r } { \lVert f - \hat { f } \rVert _ { L _ { \mu } ^ { p } ( \mathbb { R } ^ { d } ) } ^ { p } \lesssim \lVert f - \hat { f } \rVert _ { L _ { \varrho } ^ { p } ( \mathbb { R } ^ { d } ) } ^ { p } + D ( \mu , \varrho ) , } \end{array}\tag{1.1}
$$

assuming some mild regularity of $f , { \hat { f } } ,$ such as Lipschitz continuity. Here and elsewhere, we write $A \lesssim B$ to mean there exists a numerical constant $c > 0$ such that $A \leq c B$ , and likewise for $A \gtrsim B$ . Such an estimate bounds the OOD error in terms of the in-distribution error plus a term $D ( \mu , \varrho )$ measuring the distance between the two measures. Often, $D = W _ { p }$ is the Wasserstein p-distance. For instance, when $p = 2$ , a typical bound is

$$
\begin{array} { r } { \bigl \| \boldsymbol { f } - \hat { \boldsymbol { f } } \bigr \| _ { L _ { \mu } ^ { 2 } (  { \mathbb { R } } ^ { d } ) } ^ { 2 } \leq \| \boldsymbol { f } - \hat { \boldsymbol { f } } \| _ { L _ { \varrho } ^ { 2 } (  { \mathbb { R } } ^ { d } ) } ^ { 2 } + C ( \boldsymbol { f } , \hat { \boldsymbol { f } } , \varrho , \mu ) W _ { 2 } ( \mu , \varrho ) . } \end{array}\tag{1.2}
$$

Here $C ( f , \hat { f } , \varrho , \mu ) = ( L _ { f } + L _ { \hat { f } } ) \sqrt { 4 ( L _ { f } + L _ { \hat { f } } ) ^ { 2 } ( m _ { 2 } ( \mu ) + m _ { 2 } ( \varrho ) ) + 1 6 ( \vert f ( 0 ) \vert ^ { 2 } + \vert \hat { f } ( 0 ) \vert ^ { 2 } ) }$ $L .$ denotes the Lipschitz constant and $m _ { 2 } ( \cdot )$ denotes the second moment. Unfortunately, these error bounds are only meaningful for small distribution shifts. For example, if $\varrho = \mathcal { U } ( [ 0 , 1 ] ^ { d } )$ and $\mu = \mathcal { U } ( [ 0 , 1 + \delta ] ^ { d } )$ , then $W _ { p } ( \varrho , \mu ) \asymp \delta \sqrt { d }$ . Hence these bounds become meaningless unless δ is small. More generally, convergence of the indistribution error to zero does not imply the same for the OOD error in the presence of a fixed discrepancy $D ( \mu , \varrho )$

However, in practice, ML models often generalize much better than such bounds suggest. To illustrate, in Fig. 1 we show in-distribution and OOD generalization errors for function approximation and operator learning tasks. In the both cases, the terms $W _ { 2 } ( \varrho , \mu )$ are large. Yet, the OOD error decays as $m  \infty$ , albeit at a slower rate than the in-distribution error (yet still algebraic). Our aim in this paper is to explain this phenomenon.

1.2. Main results. Our results identify classes of functions and operators for which the OOD error converges algebraically for arbitrary measures supported in certain explicitly quantified domains. We now describe our setup for functions, before considering operators later.

In- and out-of-distribution measures. For reasons we discuss in $\ S 2$ , we consider functions $f : \mathbb { R } ^ { \mathbb { N } } \to \mathbb { R }$ of infinitely-many variables. However, our results apply seamlessly to functions of $d < \infty$ variables (Remark 2.3). For the training distribution, we consider the uniform probability measure $\varrho = \mathcal { U } ( [ - 1 , 1 ] ^ { \mathbb { N } } )$ . Now let $\omega = ( \omega _ { i } ) _ { i \in \mathbb { N } } \geq \mathbf { 1 }$ (understood componentwise). For the test distribution, we consider any probability measure $\mu$ with supp $( \mu ) \subseteq D _ { w }$ , where

![](images/3b4e2eee13524c6d242a2e6795a927186acb0b34f1fcc23cde6332d65f785f30.jpg)

![](images/8c3a3386632182e87cf7826db796c73d4d6d30e8fe06c4329cf324d267023ff8.jpg)  
Fig. 1: For both function (left) and operator (right) learning tasks, the OOD generalization error decays algebraically even though $W _ { 2 } ( \varrho , \mu )$ is large. Top row: the squared relative $L _ { \mu ^ { - } } ^ { 2 }$ norm error versus number of training samples m for various test distributions $\mu = \mu _ { \theta }$ . Here $\theta$ is a parameter that controls how far the test distribution is from the training distribution, which corresponds to the case $\theta = 0$ . Bottom row: the distance $W _ { 2 } ( \varrho , \mu )$ between the training and distributions $\varrho$ and $\mu .$ . This experiment considers the wing weight function (left), learned with a fully-connected DNN, and the solution operator of the Navier–Stokes PDE (right), learned with a so-called Fourier neural operator. See §6.2 and §6.3, respectively, for further details and discussion on these experiments.

$$
D _ { \omega } = [ - \omega _ { 1 } , \omega _ { 1 } ] \times [ - \omega _ { 2 } , \omega _ { 2 } ] \times \cdot \cdot \cdot .\tag{1.3}
$$

Training data and learning problem. We consider the standard regression setting in ML, where $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { m } \sim _ { \mathrm { i . i . d } } $ $\varrho$ and we are given noisy training data $( \pmb { x } _ { i } , y _ { i } : = f ( \pmb { x } _ { i } ) + e _ { i } ) , i = 1 , \ldots , m$ . Our results consider a bounded noise model, which may be adversarial, whose contribution is controlled by $\lVert e \rVert _ { 2 }$ . Now consider a model class ${ \mathcal { M } } - \mathrm { i . e . , }$ a set of functions $\mathbb { R } ^ { \mathbb { N } } \to \mathbb { R }$ , taken in this work to be family of polynomials or DNNs – and the estimator

$$
\hat { f } \in \underset { g \in \mathcal { M } } { \mathrm { a r g m i n } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } | g ( \pmb { x } _ { i } ) - y _ { i } | ^ { 2 } .\tag{1.4}
$$

Class of functions. Given $\varepsilon > 0$ and $\pmb { b } \in \mathcal { l } ^ { 1 } ( \mathbb { N } )$ with $\begin{array} { r } { b \geq 0 } \end{array}$ , we consider the class $\mathcal { H } ( b , \varepsilon )$ of $( b , \varepsilon )$ -holomorphic functions (with norm at most one). As we discuss further in §1.3-1.4, this is an important class in high-dimensional function and operator learning.

Extrapolation condition. As noted, our results employ a multiplicative OOD error bound which allows for large distribution shifts, with size determined by regularity of the functions being learned. To state this condition, we require several further concepts. First, we define the monotone ℓ<sup>p</sup>-space, $0 < p \le \infty$ , denoted $\ell _ { \mathsf { M } } ^ { p } ( \mathbb { N } )$ , as the space of all sequences $\pmb { z } = ( z _ { i } ) _ { i \in \mathbb { N } }$ whose minimal monotone majorant $\tilde { z } = ( \tilde { z } _ { i } ) _ { i \in \mathbb { N } } \in \ell ^ { p } ( \mathbb { N } )$ , where $\tilde { z } _ { i } : = \operatorname* { s u p } _ { j \geq i } | z _ { j } |$ . Second, we define $\zeta : [ 1 , \infty )  [ 1 , \infty )$ as $\zeta ( \omega ) = \omega + \sqrt { \omega ^ { 2 } - 1 }$ . We now introduce the following condition relating $\mathbf { \delta } _ { b , \ \omega }$ and a further scalar $p \in ( 0 , 1 )$ :

$$
b \odot r _ { p } \in \ell _ { \mathsf { M } } ^ { p } ( \mathbb { N } ) , \quad \left\| b \odot ( r _ { p } - \mathbf { 1 } ) \right\| _ { 1 } < \varepsilon , \qquad \mathrm { w h e r e } \ r _ { p } = ( \zeta ( \omega _ { i } ) ^ { 2 / p - 1 } ) _ { i \in \mathbb { N } } .\tag{1.5}
$$

Here $\odot$ is the Hadamard product. The first condition is a decay requirement on the sequence b. The second limits the enlargement of the domain using the holomorphy budget ε. As we see below, smaller values of $p$ give faster convergence rates, but increase $\boldsymbol { r } _ { p }$ and therefore strengthen both requirements. We discuss concrete examples of this condition in §4.3.

Theorem 1.1 (Theorem 4.1, informal version). Let $\omega \geq { \bf 1 }$ and $D _ { \omega }$ be as in (1.3). Then there exists a polynomial model class $\mathcal { M } = \mathcal { P }$ with the following property. Let $f \in { \mathcal { H } } ( b , \varepsilon )$ , where b is such that (1.5) holds for some $0 < p < 1$ . Then any minimizer $\hat { f } \ o f \ ( 1 . 4 )$ satisfies

$$
\| f - \hat { f } \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { N } ) } \lesssim C ( b , \varepsilon , \omega , p ) ( m / \log ^ { 4 } ( m ) ) ^ { 1 - 1 / p } + \| e \| _ { 2 } / \log ^ { 2 } ( m )
$$

with high probability, for any measure $\mu$ supported in $D _ { \omega }$ . Moreover, all elements of $\mathcal { P }$ are polynomials with at most $m / \log ^ { 4 } ( m )$ terms in the Legendre basis.

Theorem 1.2 (Theorem 4.4, informal version). Consider the setup of the previous theorem, where $\omega _ { \mathrm { m i n } } = \mathrm { m i n } \{ \omega _ { i } \} > 1$ . Then there exists a family of feedforward tanh DNNs $\mathcal { M } = \mathcal { N }$ achieving the same bound, up to an additional additive term proportional to $2 ^ { - m }$ . Moreover width $. ( \mathcal { N } ) \ \lesssim \ m / \log ( \omega _ { \mathrm { m i n } } )$ and $\mathrm { d e p t h } ( \mathcal { N } ) \ \lesssim$ $\log ( \log ( m ) / \log ( \omega _ { \mathrm { m i n } } ) )$

Finally, we also consider operator learning [16, 18, 44, 74], where the target is an operator $F : \mathcal { X }  \mathcal { Y }$ between two separable Hilbert spaces. We consider an indistribution probability measure ν on $x ,$ , an OOD measure $\mu$ on $x ,$ training data $( X _ { i } , Y _ { i } : = F ( X _ { i } ) + E _ { i } ) , \ i \ = \ 1 , \ldots , m ,$ where $X _ { i } ~ \sim _ { \mathrm { i . i . d } }$ $\nu ,$ a class of holomorphic operators $\mathcal { H } ( \boldsymbol { b } , \varepsilon ; \mathcal { X } , \mathcal { Y } )$ (see Definition 5.4), a family N of neural operators $\mathcal { X }  \mathcal { V }$ and an estimator $\begin{array} { r } { \hat { F } \in \underset { \mathcal { G } \in { \cal N } } { \mathrm { a r g m i n } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left. \boldsymbol { G } ( \boldsymbol { X } ) - Y _ { i } \right. ^ { 2 } } \end{array}$

Theorem 1.3 (Theorem 5.6; informal version). Let $\omega \mathrm { ~ > ~ } 1$ , where $\omega _ { \mathrm { m i n } } ~ =$ min $\left\{ \omega _ { i } \right\} > 1$ , and suppose that Assumptions 5.1-5.3 hold. Then there is a class $o f D N O s { \mathcal { N } }$ such that, for any $F \in \mathcal H ( b , \varepsilon ; \mathcal { X } , \mathcal { Y } )$ and b satisfying (1.5), the estimator $\hat { F }$ achieves a similar bound with high probability. Moreover, width $( \mathcal { N } ) \lesssim m / \log ( \omega _ { \mathrm { m i n } } )$ and $\mathrm { d e p t h } ( \mathcal { N } ) \lesssim \log ( \log ( m ) / \log ( \omega _ { \mathrm { m i n } } ) )$

See §5 for full details on this case. This theorem asserts a similar OOD bound for holomorphic operators to that given in Theorem 1.2 for holomorphic functions.

## 1.3. Contributions. We now discuss several key features of this work.

(i) Practical relevance, unknown anisotropy. Holomorphic functions and operators arise in many parametric PDE and operator learning problems. See §1.4 for further discussion. Their anisotropic regularity permits algebraic approximation rates even in infinite dimensions. A notable feature of our results is that the training problem (i.e., the model class M and loss function) are independent of b. They are consequently applicable to the unknown anisotropy setting, where the estimator has no a priori knowledge of the smoothness of the target function or operator. As discussed in [2, §3.3], this is particularly relevant in applications, where the target may be a black-box.

(ii) Distribution-agnostic bounds. Our $L ^ { \infty }$ -bounds hold uniformly over test distributions $\mu$ supported in a prescribed extrapolation domain $D _ { \omega }$ . The model class may depend on $\omega .$ , but it requires no further knowledge of $\mu .$ . Notably, our bounds require neither a bounded density $\mathrm { d } \mu / \mathrm { d } \varrho$ nor proximity of $\mu$ to $\varrho$ in some metric. In particular, $\mu$ may be singular with respect to $\varrho$ and place all its mass outside the training support. This is a major conceptual departure of our work from standard theory (recall §1.1).

(iii) Quantifiable extrapolation. (1.5) quantifies how far one can extrapolate while retaining algebraic convergence. For example, suppose that $\omega _ { i } = \omega , \forall i \in \mathbb { N }$ Then (1.5) holds whenever $\pmb { b } \in \ell _ { \mathsf { M } } ^ { p } ( \mathbb { N } )$ and $( \zeta ( \omega ) ^ { 2 / p - 1 } - 1 ) \| \pmb { b } \| _ { 1 } < \varepsilon$ and yields the noise-free convergence rate $( m / \log ^ { 4 } ( m ) ) ^ { 1 - 1 / p }$ . Thus, one can extrapolate to arbitrary accuracy as $m  \infty$ to hypercubes whose side lengths are a constant size larger than the in-distribution domain $[ - 1 , 1 ] ^ { \mathbb { N } }$ . This is very diferent to bounds of §1.1, where the discrepancy $D ( \mu , \varrho )$ limits the accuracy. But (1.5) also allows for more. Assuming suficient decay of the $b _ { i }$ , it permits domains where $\omega _ { i } \to \infty$ as $i \to \infty$ . For example, suppose that $b _ { i } = c i ^ { - a }$ and $\omega _ { i } = 1 + i ^ { b }$ for $a , b , c > 0$ satisfying (1.5). For $c > 0$ suficiently small, our results give rates arbitrarily close to $( m / \log ^ { 4 } ( { m } ) ) ^ { 1 - ( a + b ) / ( 1 + 2 b ) }$ This example illustrates that one can extrapolate arbitrarily far in the ith variable as $i \to \infty$ . Such a conclusion is, on the face of it, quite remarkable, and very unintuitive from the perspective of distribution shift theory. See §4.3 for further discussion.

(iv) Stability and a blessing of high dimensionality. In low dimensions, polynomial approximations to holomorphic functions achieve in-distribution errors that decay exponentially. Exponential rates can be retained for extrapolation, but at the cost of exponential instability. Such instability can be avoided, but with a dramatic reduction in convergence order from exponential to algebraic. By contrast, natural in-distribution rates in infinite dimensions are algebraic. Our results in (iii) reveal a blessing of high dimensionality. Stable extrapolation to admissible hypercubes preserves the same algebraic convergence order as the in-distribution rates. Moreover, algebraic rates, albeit slower, persist for certain hyperrectangles with unbounded side lengths. See Remark 3.3 for further details.

(v) Sample eficiency and optimal rates. Problems in SciML are often data-starved, making it critical to establish results on the sample complexity of learning tasks. We consider standard i.i.d. samples from the underlying measure $\varrho .$ In particular, the samples, being independent of $\omega ,$ do not need to be adapted to the extrapolation domain. Our results show that holomorphic functions and operators can be learned in a sample-eficient manner with algebraic rates in terms of $m$ , which are also optimal in various settings (see Remark 4.7).

(vi) Generalization to rougher inputs. A crucial consideration in operator learning is generalization to rougher inputs (i.e., functions of lower regularity) than those seen in training [27,43,51,74,85]. Fig. 1 already shows an example of this efect, as the eigenvalues of covariance operator of the test distributions decay more slowly than those of the training distribution (see §6.3 for further examples). Theorem 1.3 shows this is possible: the learned model can generalize to test distributions $\mu$ that are arbitrarily rough, and do so in a quantifiable manner, given suficient smoothness of the target operator. See §5.3 for further discussion.

## 1.4. Related work. We now discuss various related areas of work.

The challenge of OOD generalization. Practical demands mean that SciML models are often applied to inputs drawn from distributions difering significantly from their training distribution. Many recent works have discussed the importance and challenges of OOD generalization [14,17,19,23,43,56,62,63,70,71,77]. More broadly, OOD generalization is a critical issue in AI for Science [84], and a key component of the robustness of AI models.

Understanding and enhancing the OOD performance is also a large topic in ML in general, with well-established concepts and techniques such as domain adaptation and generalization, distributionally robust optimization, invariant risk minimization, risk extrapolation and others. See, e.g., [7,11,46,58,63,80,83]. Much of the literature focuses on classification, as opposed to regression problems, making it not directly applicable to SciML [83]. Further, distribution shifts in SciML are typically highly structured [71], which renders concepts and approaches, including the distribution shift bounds discussed in §1.1, potentially less relevant.

Existing OOD works. Distributional shift bounds of the form (1.1) can be found in many places in the ML literature (see [10, 37, 42, 51] and references therein). Use of the $W _ { p }$ -distance is standard [12, 37, 42, 47, 51–53, 72] and the specific estimate (1.2) is from [37, Prop. 3.1], which is based on [42, 51]. See [10] for similar bounds using the TV distance and so-called H-divergences. Our results contrast with these general-purpose additive bounds, as they rely on a multiplicative bound (Lemma 3.1). They are also distribution-free, as they depend only on the support. A distributionfree bound was also developed in [41, Cor. 5.2] using the modulus of continuity. But it is also additive in nature, thus only relevant to small shifts.

OOD generalization in SciML has been investigated, primarily empirically, in many works [13, 17, 23, 33, 51, 54, 62, 70, 71, 71, 83, 85]. Several of these demonstrate successful generalization under substantial shifts [17,27,63,71,85], a phenomenon our work strives to explain. Among theoretical contributions, [51] derives OOD bounds for neural operators applied to high-frequency PDE problems. Yet these employ the bounds of §1.1, and are therefore valid only for small shifts. More closely related to our work, [27] establishes algebraic convergence rates for Bayesian learning of linear operators. Our work can be viewed a generalization from linear operators to holomorphic operators, albeit in a non-Bayesian context. The work [80] also shares some similarity with ours, since it considers genuine extrapolation versus small shifts. But it does not provide generalization error bounds for concrete function classes.

Assumptions and technical ingredients. Infinite-dimensional homolorphy was developed extensively in the study of parametric DEs. It is now known that many families of parametric DEs have solution maps that are $( b , \varepsilon )$ -holomorphic functions. The list includes parametric difusion equations, parametric heat equations, PDEs over parametrized domains and various parametric initial value problems. See [3, 4, 21, 24, 64, 65, 68] and references therein. Holomorphic operators have also received substantial attention in recent years. See [5,40,44,67,79] and references therein. Besides their relevance to applications, holomorphic operators represent one of the few known classes of operators that are not subject to the curse of parametric complexity or curse of sample complexity, in contrast to finite-regularity operators [6, 50]. However, virtually all theory on learning holomorphic functions or operators considers in-distribution performance. Our work extends this to the OOD regime.

Besides holomorphy, our work has two further technical ingredients. First, modern polynomial approximation theory of infinite-dimensional functions. See, e.g., [2, 3, 21, 24, 25] and references therein. Second, the powerful technique of emulation of polynomials via DNNs/DNOs [38, 59, 66, 68, 82] (see [2, §7.1] for a review). Note that emulation techniques have been at the core of the modern approximation theory of DNNs/DNOs which has developed in the last decade [29, 32]. While emulation techniques are normally employed to show the DNNs/DNOs with certain approximation guarantees, in this work we follow ideas first developed in [1] to employ them to establish full generalization bounds.

Relation to classical analytic continuation. Extrapolation of holomorphic functions is synonymous with analytic continuation, which is a classical topic in numerical analysis (see [76] and references therein). The work [28] considered analytic continuation of univariate functions via polynomials and linear least squares. Our results can be seen as a generalization of [28] to multivariate functions. In [76] it is shown that univariate analytic continuation is ill-posed without further constraints, such as boundedness. The class $\mathcal { H } ( b , \varepsilon )$ consists of unit norm, and therefore bounded, functions and hence avoids this problem. But [76] shows that analytic continuation is always ill-conditioned with respect to $\ell ^ { \infty }$ -norm perturbations of the data, with infinite condition number. Our results do not contradict this statement, as they only show ℓ<sup>2</sup>-norm stability via the term $\lVert e \rVert _ { 2 }$ . See Remark 4.8 for further discussion.

1.5. Limitations. Holomorphy is a strong assumption. Although relevant to many SciML problems, other problems lack this regularity. Indeed, there is a growing discussion in operator learning as to what are the right structures beyond regularity that may govern learnability [16, 18, 44, 74]. Our work does not strive to address this question, which is largely open even in the in-distribution context (see also §7). On a related note, the fact that our DNN/DNO results use emulation means that the resulting estimators perform no better than corresponding polynomial estimators. Our results also employ handcrafted DNN/DNO model classes, which are some ways away from those typically used in practice (see also §7). Nonetheless, our work aligns with a recent trend in operator learning, in which more classical approaches such as polynomials [79] and kernel methods [9] have been revisited and generalized to the operator setting, and shown to be competitive with DNOs in terms of in-distribution performance. Our theoretical results provide similar conclusions for OOD performance.

There are also many diferent types of OOD generalization in SciML. In operator learning, for instance, there is much interest in multi-operator learning and foundation models, i.e., DNOs that can simultaneously solve multiple PDEs [19, 22, 39, 43, 71, 81, 83]. This is a type of OOD generalization where one wants to generalize to PDEs not seen in training. While not unrelated, it difers from our setting, as we consider learning a single operator only. Our setting is also diferent from extrapolation-in-time, as is common in neural PDE solvers [33, 43, 78]. Another distinct type of extrapolation is superresolution, where one trains a DNO on a coarse grid, then uses finer grids at test time [55] (see also [73] and references therein).

Finally, we stress that the objective of this work is to understand the OOD generalization of estimators, rather than methods of enhancing it. We discuss this important topic further in $\ S 7 .$

1.6. Outline. In §2 we introduce the class of functions considered and discuss their approximation via polynomials. In §3 we begin our focus on OOD errors, by deriving a key multiplicative bound which underpins all our main results. Our main results are presented in $\ S 4$ (for functions) and $\ S 5$ (for operators). In §6 we present numerical experiments and in $\ S 7$ we conclude and discuss open problems. Proofs of all results in this paper can be found in §SM1, while §SM2 contains additional details on the numerical experiments.

2. Holomorphic functions and polynomial approximation. As mentioned in §1.2, we consider functions $f : \mathbb { R } ^ { \mathbb { N } } \to \mathbb { R }$ and the in-distribution measure $\varrho =$ $\mathcal { U } ( [ - 1 , 1 ] ^ { \mathbb { N } } )$ . However, our theory applies seamlessly to functions of finitely-many variables (Remark 2.3). Working in infinite dimensions is convenient theoretically and practically relevant, even for finite-dimensional functions, as finite-dimensional functions behave like infinite-dimensional functions in terms of their approximation theory as soon as the dimension is moderately large (Remark 2.5).

2.1. Polynomial expansions and s-term approximation. We first recap some standard notions from polynomial approximation. See, e.g., [3, 24]. Let $P _ { n } .$ $n \in  { \mathbb { N } } _ { 0 }$ , denote the Legendre polynomials on $[ - 1 , 1 ]$ , with normalization $P _ { n } ( 1 ) = 1$ We define the normalized polynomials

$$
\psi _ { n } ( x ) = \sqrt { 2 n + 1 } P _ { n } ( x ) , \quad \forall n \in \mathbb { N } _ { 0 }\tag{2.1}
$$

and note that these form an orthonormal basis with respect to the uniform probability measure on $[ - 1 , 1 ]$ . To extend to $\mathbb { R } ^ { \mathbb { N } }$ , we proceed by tensorization. Let $\pmb { n } = ( n _ { 1 } , n _ { 2 } , . . . ) \in \mathbb { N } _ { 0 } ^ { \mathbb { N } }$ and write s $\mathrm { u p p } ( n ) \ = \ \{ i \ : \ n _ { i } \neq 0 \} \ \subseteq \mathbb { N }$ for its support. We define $\mathcal { F } = \{ \pmb { n } \in \mathbb { N } _ { 0 } ^ { \infty } : | \mathrm { s u p p } ( \pmb { n } ) | < \infty \}$ and consider the multivariate polynomials $\begin{array} { r } { \Psi _ { n } ( \pmb { x } ) = \prod _ { i \in \mathbb { N } } \psi _ { n _ { i } } ( x _ { i } ) } \end{array}$ . The set $\{ \Psi _ { n } \} _ { n \in \mathcal { F } }$ forms an orthonormal basis of $\bar { L } _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathbb { N } } )$ and therefore any $f \in L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathbb { N } } )$ has the convergent expansion

$$
f = \sum _ { n \in \mathcal { F } } c _ { n } \Psi _ { n } , \qquad \mathrm { w h e r e ~ } c _ { n } = \langle f , \Psi _ { n } \rangle _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { n } ) } = \int _ { \mathbb { R } ^ { n } } f ( \boldsymbol { x } ) \overline { { \Psi _ { n } ( \boldsymbol { x } ) } } \mathrm { d } \varrho ( \boldsymbol { x } ) .\tag{2.2}
$$

We are interested in n-term approximations to such functions. An n-term approximation to f based on an multi-index set $S \subset { \mathcal { F } } , | S | = n$ , has the form

$$
f _ { S } = \sum _ { n \in S } c _ { n } \Psi _ { n } \in \mathcal { P } _ { S } , \qquad \mathrm { ~ w h e r e ~ } \mathcal { P } _ { S } = \operatorname { s p a n } \{ \Psi _ { n } : n \in S \} .\tag{2.3}
$$

2.2. Infinite-dimensional holomorphic functions. We now introduce the class of holomorphic functions considered. In the univariate setting, it is well known that the expansion of a function $f$ using the first n Legendre polynomials converges with rate $\rho ^ { - n }$ , where $\rho \geq 1$ is the parameter of the largest Bernstein ellipse $\mathcal { E } _ { \rho }$ within which f is holomorphic. Here

$$
\begin{array} { r } { \mathcal { E } _ { 1 } = [ - 1 , 1 ] , \qquad \mathcal { E } _ { \rho } = \big \{ ( z + z ^ { - 1 } ) / 2 : z \in \mathbb { C } , ~ 1 \leq | z | \leq \rho \big \} \subset \mathbb { C } , \quad \rho > 1 . } \end{array}
$$

Generalizing via tensor products, we define the Bernstein polyellipse of parameter $\rho = ( \rho _ { 1 } , \rho _ { 2 } , . . . ) \in [ 1 , \infty ) ^ { \mathbb { N } } { \mathrm { ~ a s ~ } } \mathcal { E } _ { \rho } = \mathcal { E } _ { \rho _ { 1 } } \times \mathcal { E } _ { \rho _ { 2 } } \times \dots \subset \mathbb { C } ^ { \mathbb { N } }$

Definition 2.1 ((b, ε)-holomorphy). Let $\pmb { b } \in [ 0 , \infty ) ^ { \mathbb { N } }$ and $\varepsilon > 0$ . A function $f : \mathbb { R } ^ { \mathbb { N } } \to \mathbb { C }$ is (b, ε)-holomorphic if it is holomorphic in every Bernstein polyellipse $\mathcal { E } _ { \rho }$ with parameter $\pmb { \rho } = ( \rho _ { i } ) _ { i = 1 } ^ { \infty } \in [ 1 , \infty ) ^ { \mathbb { N } }$ satisfying

$$
\sum _ { i = 1 } ^ { \infty } \left( ( \rho _ { i } + \rho _ { i } ^ { - 1 } ) / 2 - 1 \right) b _ { i } \leq \varepsilon .\tag{2.4}
$$

See $\ S 1 . 4$ for relevant literature on this class. For convenience we write

$$
\mathcal { R } _ { b , \varepsilon } = \bigcup \{ \mathcal { E } _ { \rho } : \rho \in [ 1 , \infty ) ^ { \mathbb { N } } , \ \rho \ \mathrm { s a t i s f i e s } \ ( 2 . 4 ) \} \subseteq \mathbb { C } ^ { \mathbb { N } }
$$

for the region and $\mathcal { H } ( \pmb { b } , \varepsilon ) = \{ f : D  \mathbb { C } \ ( \pmb { b } , \varepsilon )$ -holomorphic, $\| f \| _ { L ^ { \infty } ( \mathcal { R } _ { b , \epsilon } ) } \le 1 \}$ for the set of functions that are holomorphic in $\mathcal { R } _ { b , \varepsilon }$ with uniform norm at most one.

Note that the sequence $^ { b }$ determines the type of anisotropic behaviour of functions in $\mathcal { H } ( b , \varepsilon )$ . Indeed, if $b _ { j }$ is large for some $j \in \mathbb N$ , then (2.4) holds only for small values of $\rho _ { j }$ , meaning that $f$ is less smooth with respect to the variable $x _ { j }$ . Conversely, if $b _ { j }$ is small (or even $b _ { j } = 0 )$ , then $f$ is more smooth (entire) in the variable $x _ { j }$

Remark 2.2 (Unknown anisotropy). In some cases, suitable parameters $( b , \varepsilon )$ may be known, e.g., via a priori analysis of a given problem. However, in practical cases, values for these parameters are typically unknown. We refer to this as the unknown anisotropy setting. Our main focus is on this more practical case, with our goal being to design estimators with guaranteed convergence rates that are independent of $( b , \varepsilon )$ . See [2, §3.3] for more discussion on known versus unknown anisotropy.

Remark 2.3 (Finite-dimensional functions). Definition 2.1 is somewhat complicated, in that it requires the function to have a holomorphic extension to a union of Bernstein polyellipses. This is used in infinite dimensions to obtain algebraic rates of convergence of the best n-term approximation. In finite dimensions, it is enough for the function to be holomorphic in a single Bernstein polyellipse. Concretely, let $f :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ be holomorphic in the finite-dimensional Bernstein polyellipse $\mathcal { E } _ { \bar { \rho } _ { 1 } } \times \cdot \cdot \cdot \times \mathcal { E } _ { \bar { \rho } _ { d } } \subset \mathbb { C } ^ { d }$ . Now let

$$
b _ { i } = \varepsilon \left( ( \bar { \rho } _ { i } + \bar { \rho } _ { i } ^ { - 1 } ) / 2 - 1 \right) ^ { - 1 } , \ i \in [ d ] , \qquad b _ { i } = 0 , \ i \in \mathbb { N } \backslash [ d ] .
$$

Then the natural extension $\tilde { f }$ of $f$ to a function of infinitely-many variables, defined by $\tilde { f } ( \pmb { x } ) = f ( x _ { 1 } , . . . , x _ { d } )$ for all ${ \pmb x } = ( x _ { i } ) _ { i \in \mathbb { N } } \in \mathbb { R } ^ { \mathbb { N } }$ , is $( b , \varepsilon )$ -holomorphic. Hence, all results that follow also apply seamlessly to finite-dimensional holomorphic functions.

2.3. Best n-term approximation of holomorphic functions. Best n-term approximation is a type of nonlinear approximation [31] in which one chooses the subset $S \subset { \mathcal { F } } , | S | = n$ , that minimizes the error $f - f _ { S }$ in a certain norm, where $f _ { S }$ is the corresponding approximation $( 2 . 3 )$ . We shall consider this with respect to the L<sup>2</sup>-norm (although other L<sup>p</sup>-norms, $p \neq 2$ , could equally be considered). We define an $L _ { \varrho } ^ { 2 } .$ -norm best n-term approximation as

$$
f _ { n } = f _ { S ^ { * } } , \qquad \mathrm { w h e r e ~ } S ^ { * } \in \mathrm { a r g m i n } \{ \| f - f _ { S } \| _ { L _ { o } ^ { 2 } ( \mathbb { R } ^ { N } ) } : S \subset \mathcal { F } , ~ | S | = n \} .
$$

Due to Parseval’s identity, $S ^ { * }$ has an explicit characterization as the index set corresponding to n largest coeficients of $f$ in absolute value (note that $S ^ { * }$ , and therefore $f _ { n }$ , need not be unique, due to ties, but this makes no diference in what follows). The following result is well known (see, e.g., [3, Thm. 3.28] or [24, §3.2]). It demonstrates that the best n-term approximation of any function in $\mathcal { H } ( b , \varepsilon )$ converges with algebraic rate whenever $^ { b }$ is $\ell ^ { p } { \mathrm { - s u m m a b l e } }$

Theorem 2.4 (Algebraic convergence of the best s-term approximation). Let $\varepsilon > 0$ and $\pmb { b } \in [ 0 , \dot { \infty } ) ^ { \mathbb { N } }$ be such that $\pmb { b } \in \mathcal { l } ^ { p } ( \mathbb { N } )$ for some $0 < p < 1$ . Then, for any $2 \leq q \leq \infty$

$$
\left\| f - f _ { n } \right\| _ { L _ { \varrho } ^ { q } ( \mathbb { R } ^ { n } ) } \leq C ( \pmb { b } , \varepsilon , p ) \cdot n ^ { 1 - \frac { 1 } { q } - \frac { 1 } { p } } , \qquad \forall f \in \mathcal { H } ( \pmb { b } , \varepsilon ) , \ n \in \mathbb { N } .\tag{2.5}
$$

This remarkable result states that there are classes of functions whose approximation by polynomials is free from the curse of dimensionality, despite depending on infinitely-many variables. Recall that b controls the anisotropy of functions in $\mathcal { H } ( b , \varepsilon )$ The faster $^ { b }$ decays $( { \mathrm { i . e . } }$ , the smaller $p )$ , the more anisotropic the functions and the faster the rate in (2.5).

Remark 2.5 (Finite-dimensional functions and exponential rates). Standard polynomial approximation theory asserts that the n-term approximation of a function of $d < \infty$ variables that is holomorphic in $\mathcal { E } _ { \rho _ { 1 } } \times \cdots \times \mathcal { E } _ { \rho _ { d } }$ converges exponentially fast in $n ^ { 1 / d }$ . While exponential, this rate exhibits the curse of dimensionality, and raises the question: what happens for large $d \mathrm { ? }$ The bound (2.5) provides an answer. By assuming holomorphy and anisotropy, one achieves d-independent algebraic rates (here we recall that (2.5) also holds for any holomorphic function of $d$ variables, due to Remark 2.3). Typically, the error follows the algebraic rate for small $n$ , before transitioning to the exponential rate as n grows larger. However, when $d \approx 1 0$ or larger, this transition point typically occurs at a large value of $n _ { \mathrm { : } }$ meaning that the exponential rates are often not witnessed in practical computations [2]. Thus, algebraic rates are the relevant even in finite dimensions, except when d is small. See §6.1 for several examples of this phenomenon.

3. From in-distribution to out-of-distribution. The bound (2.5) considers only the in-distribution error in the $L _ { \varrho } ^ { q } \mathrm { - n o r m }$ . Further, it only asserts the existence of a polynomial achieving the specified algebraic rate, but no insight into how to compute from data. In this section, we focus on the OOD error, for an arbitrary measure $\mu .$ We introduce an OOD generalization guarantee which, in contrast to those discussed in §1.1, is multiplicative rather than additive. We then apply this to the polynomial setting in Theorem 3.2, demonstrating that it guarantees good OOD generalization up to the size of the in-distribution error.

3.1. A multiplicative OOD generalization guarantee. Let M be an arbitrary model class or hypothesis set for the regression problem, i.e., a set of functions $\mathbb { R } ^ { \mathbb { N } } \to \mathbb { R }$ from which we aim to compute an estimator $\hat { f } \in \mathcal { M }$ to an unknown function $f$ . Given two measures $\varrho , \mu$ on $\mathbb { R } ^ { \mathbb { N } }$ , we define the distribution shift constant of $\mathcal { M }$ as

$$
\Delta ( \mathcal { M } ; \varrho , \mu ) = \operatorname* { s u p } \left\{ \frac { \| g _ { 1 } - g _ { 2 } \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { \mathbb { N } } ) } } { \| g _ { 1 } - g _ { 2 } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathbb { N } } ) } } : g _ { 1 } , g _ { 2 } \in \mathcal { M } , \ g _ { 1 } \neq g _ { 2 } \right\} .\tag{3.1}
$$

This constant measures the worst-case OOD behaviour of the diference of two elements of M relative to their in-distribution behaviour. Note that it is possible to formulate $\Delta ( \mathcal { M } ; \varrho , \mu )$ in terms of other norms. We use the $L _ { \varrho } ^ { 2 } \mathrm { - n o r m }$ as we generally consider estimators that are defined through empirical least-squares fits. We use the $L _ { \mu } ^ { \infty }$ -norm in order to have error bounds that depend on the support of the measure $\mu$ only (since, in our work, all functions are continuous).

Lemma 3.1 (Multiplicative OOD generalization bound). Let M and $\Delta ( \mathcal { M } ; \varrho , \mu )$ be as above and $f \in L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { \mathbb { N } } )$ . Then, for any $\hat { f } \in \mathcal { M }$

$$
\begin{array} { r l } {  { \| f - \hat { f } \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { N } ) } \le \operatorname* { i n f } _ { g \in { \cal M } } \Big \{ \Delta ( \mathcal { M } ; \varrho , \mu ) \| f - g \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { N } ) } + \| f - g \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { N } ) } \Big \} } \quad } & { } \\ & { + \Delta ( \mathcal { M } ; \varrho , \mu ) \| f - \hat { f } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { N } ) } . } \end{array}\tag{3.2}
$$

Proof. By the triangle inequality and the definition of $\Delta ( \mathcal { M } ; \varrho , \mu )$ , we have

$$
\begin{array} { r l } & { \| f - \hat { f } \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { \mathbb { N } } ) } \leq \| f - g \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { \mathbb { N } } ) } + \| g - \hat { f } \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { \mathbb { N } } ) } } \\ & { \qquad \leq \| f - g \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { \mathbb { N } } ) } + \Delta ( \mathcal { M } ; \varrho , \mu ) \| g - \hat { f } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathbb { N } } ) } } \end{array}
$$

for any $g \in \mathcal { M }$ . We now use the triangle inequality once more to get the result.

This result implies that accurate OOD generalization is possible provided (i) the constant $\Delta ( \mathcal { M } ; \varrho , \mu )$ is not too large, (ii) the model class M contains an element $g$ that provides simultaneously good approximations to f in both the $L _ { \varrho ^ { - } } ^ { 2 }$ and $L _ { \mu } ^ { \infty } \cdot$ norms and (iii) the in-distribution generalization error is small. Further, (3.2) also implies stability: the OOD generalization behaviour of $\hat { f }$ will be robust to errors in the training data, as long as $\Delta ( \mathcal { M } ; \varrho , \mu )$ is not too large and the in-distribution generalization performance of $\hat { f }$ is robust to such errors.

3.2. Application to polynomial model classes. This begs the question: for holomorphic functions, when is it possible to achieve $( \mathrm { i } ) { - } ( \mathrm { i } \mathrm { i } ) ?$ Our next result focuses on $( \mathrm { i } ) \mathrm { - } ( \mathrm { i i } )$ , and shows the existence of a polynomial model class that achieves these objectives. The third objective, (iii), i.e., the construction of a suitable estimator, is the focus of the next section.

Theorem 3.2. Let $\omega \ge \mathbf { 1 } , \ \varrho = \mathcal { U } ( [ - 1 , 1 ] ^ { \infty } ) , \ \omega \ge \mathbf { 1 }$ and $D _ { \omega }$ be as in (1.3). Suppose that $f \in { \mathcal { H } } ( b , \varepsilon )$ , where $\mathbf { \delta } _ { b } \geq \mathbf { 0 }$ and $\varepsilon > 0$ satisfy

$$
b \odot r _ { p } \in \ell ^ { p } ( \mathbb { N } ) , \quad \left\| b \odot ( r _ { p } - 1 ) \right\| _ { 1 } < \varepsilon , \qquad w h e r e \ r _ { p } = \zeta ( \omega ) ^ { 2 / p - 1 }\tag{3.3}
$$

for some $0 < p < 1$ . Then, for every $k > 0$ , there exists a set $S \subset { \mathcal { F } }$ depending on $^ { b , }$ $\varepsilon , \omega$ and k with $| S | \le k$ such that

$$
\Delta ( \mathcal { P } _ { S } ; \varrho , \mu ) \le \sqrt { k } ,\tag{3.4}
$$

where $\mathcal { P } _ { S }$ is as in (2.3), and a $g \in { \mathcal { P } } _ { S }$ such that

$$
\Delta ( \mathcal { P } _ { S } ; \varrho , \mu ) \| f - g \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { n } ) } + \| f - g \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { n } ) } \leq C ( b , \varepsilon , \omega , p ) k ^ { 1 - 1 / p }\tag{3.5}
$$

for any probability measure µ supported in $D _ { \omega }$

This is a key result of the paper. It provides a precise criterion (3.3) under which one achieves the algebraic rate (3.5) with a distribution shift constant (3.4) growing mildly in k. Notice that the model class depends on the anisotropy parameters b and $\varepsilon ,$ which makes it unsuitable for the unknown anisotropy setting (recall (i) in §1.3). In the next section, we shall remove this requirement through a mild strengthening of (3.3), which is equivalent to (1.5).

Remark 3.3 (The blessing of high dimensionality). Let $f : \mathbb { R } \to \mathbb { R }$ be holomorphic in $\mathcal { E } _ { \rho } , n \in \mathbb { N } , \mathcal { M } = \mathbb { P } _ { n }$ be the space of polynomials of degree at most n and suppose that $\operatorname { s u p p } ( \mu ) \subseteq [ - \omega , \omega ]$ . This case was considered in depth in [28]. As shown therein, the constant $\Delta ( \mathcal { P } _ { S } ; \varrho , \mu ) \lesssim n \zeta ( \omega ) ^ { n }$ and there exists a $g \in \mathcal { M }$ such that

$$
\begin{array} { r } { \Delta ( \mathcal { P } _ { S } ; \varrho , \mu ) \| f - g \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ) } + \| f - g \| _ { L _ { \mu } ^ { \infty } } \lesssim _ { \rho , \omega } n ( \zeta ( \omega ) / \rho ) ^ { n } . } \end{array}
$$

Thus extrapolation is possible for any ω satisfying $\zeta ( \omega ) < \rho _ { ; }$ , with an exponential rate of convergence in n. But this comes at the expense of a potentially exponentially-large stability constant $\Delta ( \mathcal { P } _ { S } ; \varrho , \mu ) \lesssim n \zeta ( \omega ) ^ { n }$ . To compare with Theorem 3.2, let $k > 0$ be arbitrary and set $\begin{array} { r } { n = \lfloor \frac { \log ( k ) } { 2 \log \zeta ( \omega ) } \rfloor } \end{array}$ so that $\Delta ( \mathcal { P } _ { S } ; \varrho , \mu ) = \mathcal { O } ( \sqrt { k } )$ . Here, the symbol $\mathcal { O } ( \cdot )$ suppresses a log factor in k and ω-dependent constant (which are irrelevant to the present discussion). With this choice of $n .$ , one ensures a similar stability level. However, this reduces the convergence rate

$$
\Delta ( \mathcal { P } _ { S } ; \boldsymbol { \varrho } , \mu ) \| f - g \| _ { L _ { \boldsymbol { \varrho } } ^ { 2 } ( \mathbb { R } ) } + \| f - g \| _ { L _ { \boldsymbol { \mu } } ^ { \infty } } = \mathcal { O } \big ( k ^ { \frac { 1 } { 2 } ( 1 - \log ( \boldsymbol { \rho } ) / \log ( \zeta ( \omega ) ) ) } \big )
$$

to algebraic in k. See [28] for a similar derivation, as well as lower bounds showing that such convergence is optimal. Here lies the blessing of high dimensionality. In low (in particular, one) dimension, the in-distribution error decays exponentially fast. However, stable extrapolation can only be achieved by reducing the rate from exponential to algebraic. Conversely, in infinite dimensions, the in-distribution error decays algebraically, and stable extrapolation can be achieved with algebraic rates, albeit with a potentially smaller order.

4. Sample-eficient OOD learning of high-dimensional functions. The previous results show that there are (polynomial) model classes that control the OOD error of an estimator $\hat { f }$ up to its in-distribution error. However, they do not describe how to compute such an estimator from data and, critically, how much data is required to do so. This is the focus of the present section, which contains the main results in this paper on function approximation. In $\ S 5$ , we establish analogous results for operator learning. We now draw $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { m }$ ∼<sub>i.i.d.</sub> ϱ and consider the training data

$$
( \pmb { x } _ { i } , y _ { i } ) , \ i = 1 , \ldots , m , \qquad \mathrm { w h e r e } \ y _ { i } = f ( \pmb { x } _ { i } ) + e _ { i } ,\tag{4.1}
$$

and $e _ { i } \in \mathbb { R }$ . We then consider the $\ell ^ { 2 } { \mathrm { - } } \mathrm { l o s s }$ over some model class M, yielding the estimator $\hat { f }$ in (1.4). We first consider polynomial model classes. Next, we use emulation techniques to establish similar results for DNN model classes.

4.1. Polynomial model classes. It is possible to obtain an estimator by directly combining Theorem 3.2, which identifies a suitable polynomial space $\mathcal { P } _ { S }$ , with rather standard tools from the analysis of linear least-squares fitting with i.i.d. random samples. However, such a result is unrealistic, since the model class (and consequently the estimator) requires knowledge of the anisotropy parameters $( b , \varepsilon )$ . Our main result establishes the existence of an estimator that is independent of $( b , \varepsilon )$ , and therefore suitable for the unknown anisotropy setting described in Remark 2.2. For this, we need the following assumption, which slightly strengthens (3.3):

$$
b \odot r _ { p } \in \ell _ { \mathsf { M } } ^ { p } ( \mathbb { N } ) , \quad \left\| b \odot ( r _ { p } - \mathbf { 1 } ) \right\| _ { 1 } < \varepsilon , \qquad \mathrm { w h e r e } ~ r _ { p } = \zeta ( \omega ) ^ { 2 / p - 1 } .\tag{4.2}
$$

Recall from §1.2 that $\ell _ { \mathsf { M } } ^ { p } ( \mathbb { N } )$ is the monotone $\ell ^ { p } .$ -space.

Theorem 4.1. Let $0 < \epsilon < 1 , m \geq \bar { m }$ , where $\bar { m } = \bar { m } ( \epsilon ) \in \mathbb { N }$ depends on ϵ only, $\varrho = \mathcal { U } ( [ - 1 , 1 ] ^ { \mathbb { N } } ) , \omega \ge \mathbf { 1 }$ and $D _ { \omega }$ be as in (1.3). Then there exists a polynomial model class $\mathcal { P }$ depending on m, ω and ϵ only with the following property. Let $f \in { \mathcal { H } } ( b , \varepsilon )$ 2 where $\mathbf { \delta } _ { b } \geq \mathbf { 0 }$ and $\varepsilon > 0$ satisfy (4.2) for some $0 < p < 1$ , draw $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { m } \sim _ { \mathrm { i . i . d } } \ \varrho$ and consider (1.4) with noisy training data (4.1). Then, with probability at least $1 - \epsilon$ any minimizer $\hat { f }$ satisfies

$$
\left\| f - \hat { f } \right\| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { N } ) } \lesssim \frac { C ( b , \varepsilon , \omega , p ) } { \sqrt { \epsilon } } \left( \frac { m } { L } \right) ^ { 1 - 1 / p } + \frac { \| e \| _ { 2 } } { \sqrt { L } }
$$

for any probability measure $\mu$ supported in $D _ { \omega }$ , where $L = L ( m , \epsilon ) : = \log ^ { 4 } ( m ) +$ $\log ( 1 / \epsilon )$ and $\boldsymbol { e } = ( e _ { i } ) _ { i = 1 } ^ { m }$ . Moreover, $\hat { f } \in \mathcal { P } _ { S }$ for some S satisfying $| S | \le m / L$

Here and elsewhere, a polynomial model class P is a subset of all algebraic polynomials, $\mathrm { i . e . , ~ } \mathcal { P } \subseteq$ span $\{ x \mapsto x ^ { n } : n \in { \mathcal { F } } \}$ , where $\begin{array} { r } { \boldsymbol { x } ^ { n } = \prod _ { i \in \mathbb { N } } x _ { i } ^ { n _ { i } } } \end{array}$ for ${ \pmb x } = ( x _ { i } ) _ { i \in \mathbb { N } }$ and ${ \pmb n } = ( n _ { i } ) _ { i \in \mathbb { N } }$ . This result shows that the algebraic OOD error rate suggested by Theorem 3.2 can be obtained from m samples via a least-squares estimator, at the loss of the log term $L ( m , \epsilon )$ . Further, the OOD error of the estimator is robust to $\ell ^ { 2 } { \mathrm { - b o u n d e d } }$ noise, due to the term $\| e \| _ { 2 } / { \sqrt { L } }$

Remark 4.2 (Condition (3.3) versus (4.2)). These conditions are identical whenever the sequence $b \odot \boldsymbol { r } _ { p }$ is monotonically nonincreasing. Observe that in the indistribution setting $( \mathrm { i } . \mathrm { e } . , \omega = \mathbf { 1 } )$ , they reduce to $\pmb { b } \in \ell ^ { p } ( \mathbb { N } )$ and $\pmb { b } \in \ell _ { \mathsf { M } } ^ { p } ( \mathbb { N } )$ , respectively. It has been shown [4] that it is impossible to construct estimators that converge as $m  \infty$ for all $\pmb { b } \in \ell ^ { p } ( \mathbb { N } )$ . However, it is possible to construct an estimator for all b satisfying $\pmb { b } \in \ell _ { \mathsf { M } } ^ { p } ( \mathbb { N } )$ . Theorem 4.1 shows that it is also possible to construct an estimator in the OOD setting that converges for all b satisfying (4.2). It is currently unknown whether (4.2) can be weakened. However, as we discuss in Remark 4.7 later, under this condition the estimator developed in Theorem 4.1 yields rates that are optimal up to the log term L for specific choices of $\omega .$

Remark 4.3 (Practicality of the estimator). Theorem 4.1 is based on a sparse polynomial approximation procedure, where ${ \mathcal { P } } = \cup _ { S \in { \mathcal { S } } } { \mathcal { P } } _ { S }$ is a union of subspaces of dimension at most $m / L ( m , \epsilon )$ . Roughly speaking, the collection $s$ is constructed by selecting all possible index sets S asserted by Theorem 3.2 for any $( b , \varepsilon )$ satisfying (4.2). Being a union of polynomial subspaces, it shares some similarities with the estimator of [26], although the choice of sets S is diferent. The strengthened condition (4.2) is required over (3.3) to ensure that $| S | < \infty$ . This is a necessary component of our analysis which, in general, uses compressed sensing techniques. Unfortunately, the estimator in Theorem 4.1 is generally impractical to compute, since, in principle, it would involve solving |S| linear least-squares problems and then choosing the one with the smallest value of the loss function. This is, in some senses, analogous to the classical problem of a sparse approximate solution to a linear system $A z = b$ by minimizing $\| A z - b \| _ { 2 } ^ { 2 }$ over the set of s-sparse vectors (since this set is a union of s-dimensional subspaces). In standard compressed sensing, one obtains practical algorithms by via convexification – for instance, by minimizing the LASSO penalty $\lambda \| z \| _ { 1 } + \| A z - b \| _ { 2 } ^ { 2 }$ . A similar approach could be considered here, by using a weighted ℓ<sup>1</sup>-norm and following ideas from [3]. However, this is not the main focus of this work.

4.2. Deep learning and DNN model classes. We now focus on DNN model classes. Let $\sigma : \mathbb { R }  \mathbb { R }$ be an activation function. In this work, we consider feedforward DNNs of the form

$$
N : \mathbb { R } ^ { n _ { 0 } } \to \mathbb { R } ^ { n _ { M + 1 } } , \ z \mapsto N ( z ) = \mathcal A _ { M } ( \sigma ( \mathcal A _ { M - 1 } ( \sigma ( \cdot \cdot \cdot \sigma ( \mathcal A _ { 0 } ( z ) ) \cdot \cdot \cdot ) ) ) ) ,\tag{4.3}
$$

where $\mathcal { A } _ { l } : \mathbb { R } ^ { n _ { l } }  \mathbb { R } ^ { n _ { l + 1 } }$ are afine maps, $n _ { 1 } , \dots , n _ { M }$ are the widths of the hidden layers, and $n _ { 0 }$ and $n _ { M + 1 }$ are the input and output dimensions, respectively. We define width $( N ) = \operatorname* { m a x } \{ n _ { 1 } , \dots , n _ { M } \}$ and depth $( N ) = M$ . We denote a DNN model class of the form (4.3) with a fixed architecture (i.e., fixed activation function, depth and widths) as $\mathcal { N }$ , and write width $( \mathcal N ) = \operatorname* { m a x } \{ n _ { 1 } , . . . , n _ { M } \}$ and depth $( \mathcal { N } ) = M$

Let $\mathcal { N }$ be a DNN model class as above, with input dimension $n _ { 0 } = n \in \mathbb { N }$ and output dimension $n _ { M + 1 } = 1$ . Given training data (4.1) of an unknown function $f ,$ we consider estimator (1.4) with $\mathcal { M } = \mathcal { N }$ . Notice that the variable $\pmb { x } \in \mathbb { R } ^ { \mathbb { N } }$ while the input dimension of any $N \in \mathcal N$ is $n _ { 0 } < \infty$ . We assume that N acts on the first $n _ { 0 }$ entries of x only, but to avoid unnecessary notation, we will simply write $N ( { \pmb x } )$ instead of $N ( \pmb { x } _ { [ n _ { 0 } ] } )$

Theorem 4.4. Let $0 < \epsilon < 1$ $m \geq { \bar { m } }$ , where $\bar { m } = \bar { m } ( \epsilon ) \in \mathbb { N }$ depends on ϵ only, $\varrho = \mathcal { U } ( [ - 1 , 1 ] ^ { \mathbb { N } } )$ $\omega \geq 1$ satisfy ω<sub>min</sub> := min $\left\{ \omega _ { i } \right\} > 1$ and $D _ { \omega }$ be as in (1.3). Then there exists a $D N N$ model class $\mathcal { N }$ with tanh activation functions depending on m, ω and ϵ only with the following property. Let $f \in { \mathcal { H } } ( b , \varepsilon )$ , where $\mathbf { \delta } _ { b } \geq \mathbf { 0 }$ and $\varepsilon > 0$ satisfy (4.2) for some $0 < p < 1$ , draw $\mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { m } \sim _ { \mathrm { i . i . d } } \ \varrho$ and consider the problem (1.4)

with noisy training data (4.1). Then, with probability at least $1 - \epsilon ,$ any minimizer $\hat { f }$ satisfies

$$
\big \| \ b { f } - \hat { \ b { f } } \big \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { \mathbb { N } } ) } \lesssim \frac { C ( \ b { b } , \varepsilon , \omega , p ) } { \sqrt { \epsilon } } \left( \frac { m } { L } \right) ^ { 1 - 1 / p } + \frac { \| \ b { e } \| _ { 2 } } { \sqrt { L } } + 2 ^ { - m } ,
$$

for any µ probability measure supported on $D _ { \omega }$ , where $L = L ( m , \epsilon ) : = \log ^ { 4 } ( m ) +$ $\log ( 1 / \epsilon )$ and $\boldsymbol { e } = ( e _ { i } ) _ { i = 1 } ^ { m }$ . Further, the class N satisfies

$$
n _ { 0 } \lesssim \frac { m } { L } , \qquad \mathrm { w i d t h } ( \mathcal { N } ) \lesssim \frac { m } { \log ( \omega _ { \mathrm { m i n } } ) } , \qquad \mathrm { d e p t h } ( \mathcal { N } ) \lesssim \log \left( \frac { \log ( m ) } { \log ( \omega _ { \mathrm { m i n } } ) } \right) .
$$

This result shows one can achieve the same error bounds as in Theorem 4.1, up to an extra ${ \mathcal { O } } ( 2 ^ { - m } )$ term, via deep learning with DNNs that are not excessively wide or deep. This class uses tanh activations, yet other activations could readily be considered (such as ReLU or general activations that are definable in an o-minimal sense [45], for which the emulation results employed in the proof of Theorem 4.4 hold). It also assumes that $\omega _ { \operatorname* { m i n } } > 1$ , although we believe this could be removed by using somewhat deeper DNNs.

Theorem 4.4 employs a specific ‘handcrafted’ family of DNNs. In the final layer, the weights can take arbitrary real values, but in the hidden layers, only a certain discrete set of values is allowed for the weights and biases (see the proof for the full construction). This is because its proof is based on combining Theorem 4.1 with tools from DNN emulation. Specifically, we approximately emulate the elements of the polynomial model class $\mathcal { P }$ as tanh DNNs, then show that minimizers of the DNN training problem yield approximate minimizers of the polynomial training problem. Finally, the result follows through a careful perturbation analysis to show the overall efect of the approximate emulation is at most ${ \mathcal { O } } ( 2 ^ { - m } )$ .

4.3. Examples and further discussion. We now discuss the condition (4.2) and the various rates described by the above theorems. We do this by considering the following two examples.

Example 4.5 (Constant side lengths). Suppose first that $\omega _ { 1 } = \omega _ { 2 } = \cdot \cdot \cdot = \omega > 1$ ， so that $D _ { \omega }$ is a hypercube with side lengths 2ω. Then (4.2) holds if and only $i f$

$$
\begin{array} { r } { \pmb { b } \in \ell _ { \mathsf { M } } ^ { p } ( \mathbb { N } ) \quad a n d \quad \omega < \omega _ { * } ( \pmb { b } , \varepsilon , p ) : = \zeta ^ { - 1 } \left( ( 1 + \varepsilon / \| \pmb { b } \| _ { 1 } ) ^ { \frac { p } { 2 - p } } \right) . } \end{array}\tag{4.4}
$$

Note here that $\zeta ^ { - 1 } : [ 1 , \infty ) \to [ 1 , \infty )$ is the Joukowsky map $\zeta ^ { - 1 } ( r ) = { \textstyle { \frac { 1 } { 2 } } } \left( r + r ^ { - 1 } \right)$ . Observe that the first condition $\pmb { b } \in \ell _ { \mathsf { M } } ^ { p } ( \mathbb { N } )$ implies that the in-distribution error satisfies

$$
\| f - \hat { f } \| _ { L _ { \mu } ^ { \infty } (  { \mathbb { R } ^ { \mathbb { N } } } ) } \lesssim \frac { C ( b , \varepsilon , p ) } { \sqrt { \epsilon } } \left( \frac { m } { L } \right) ^ { 1 - 1 / p } + \frac { \| e \| _ { 2 } } { \sqrt { L } }
$$

(this follows from Theorem 4.1 with $\omega = 1 )$ . Hence, Theorems 4.1 and $4 . 4$ show that one can achieve precisely the same rate for the OOD error, for any measure µ supported in $[ - \omega , \omega ] ^ { \mathbb { N } }$ . Notice that this goes significantly beyond the realm of the usual distribution shift theory described in §1.1, since such a $\mu$ need not be a small perturbation of $\varrho .$

Example 4.6 (Unbounded side lengths). Now suppose that $b _ { i } = c i ^ { - a }$ and $\omega _ { i } =$ $i ^ { b } \ f o r \ a , b , c > 0$ . Then (4.2) holds for $p \in ( 0 , 1 )$ if and only if a, b and ε satisfy

$$
\frac { 1 } { p } < \frac { a + b } { 1 + 2 b } { a n d } c \sum _ { i = 1 } ^ { \infty } i ^ { - a } \left( \zeta ( i ^ { b } ) ^ { 2 / p - 1 } - 1 \right) < \varepsilon .\tag{4.5}
$$

If this holds, the OOD error rate is $\mathcal { O } ( ( m / L ) ^ { 1 - \frac { a + b } { 1 + 2 b } + \delta } )$ for any $\delta > 0$ . Conversely, the in-distribution rate is $\mathcal { O } ( ( m / L ) ^ { 1 - a + \delta } )$ . Hence we achieve algebraic rates of decay of the OOD error for any measure supported in a hypercube whose side lengths can become arbitrarily large, albeit at a slower algebraic rate than the in-distribution error. This setting is a stark contrast to theory of small distribution shifts discussed in §1.1.

Remark 4.7 (Optimality of the rates). It is known that the in-distribution error cannot decay any faster than $m ^ { 1 - 1 / p } / \log ( m )$ in the worst case for $b \in \ell _ { \mathsf { M } } ^ { p } ( \mathbb { N } ) , \| \pmb { b } \| _ { p , \mathsf { M } } =$ 1, [5, Thm. 4.2]. This is a type of minimax lower bound, which considers the worst-case error over $\mathcal { H } ( b , \varepsilon )$ given arbitrary i.i.d. random samples and an arbitrary (nonlinear) estimator. As a result of Example 4.5, the estimators of Theorems 4.1 and 4.4 yield optimal OOD error rates (up to the log term $L )$ over measures $\mu$ supported in a domain with constant side lengths. Currently, it is unknown whether the algebraic rates in the setting of Example 4.6 are also optimal.

Remark 4.8 (Ill-conditioning and best guaranteed accuracy). Our results show that the estimator is robust to perturbations, yet, as noted in §1.4, analytic continuation in one dimension is ill-conditioned with infinite condition number. We now explain this seeming contradiction. The key point is that the former statements pertain to $\ell ^ { \infty } { \mathrm { - n o r m } }$ perturbations in the data. Suppose now that the errors $e _ { i }$ in the data satisfy a uniform bound $\| e \| _ { \infty } \leq \chi .$ , implying that $\| e \| _ { 2 } \leq \sqrt { m } \chi$ . Then, applying Theorem 4.1 with an optimal choice of m to balance the two error terms, we deduce that one can compute an $\hat { f }$ that satisfies (for a possibly diferent constant)

$$
\begin{array} { r } { \| f - \hat { f } \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { \mathbb { N } } ) } \lesssim C ( b , \varepsilon , \omega , p ) \chi ^ { \frac { 1 / p - 1 } { 1 / p - 1 / 2 } } . } \end{array}\tag{4.6}
$$

For this argument to imply that analytic continuation had a finite condition number, this bound would have had to scale like $\chi .$ . But the power in (4.6) is less than one, as indeed it must be so as not to contradict the one-dimensional result.

Viewing $\chi = \epsilon _ { \mathsf { m a c h } }$ as machine epsilon, the bound (4.6) can also be interpreted as the best guaranteed accuracy in finite-precision computations. It is interesting to note that this is some fractional power of machine epsilon that depends on the smoothness, and setting $\omega = 1$ , this is even the case for the in-distribution error. We believe this is an artefact of the estimator, which is based on the $\ell ^ { 2 } { \mathrm { - l o s s . } }$ or its analysis. Determining whether (4.6) represents the best achievable accuracy for any estimator is an interesting topic for future work.

5. Operator learning. Operator learning is an extension of the classical function approximation problem, where the target object is an operator between two, typically infinite-dimensional, spaces. As is customary, we consider operators $F : \mathcal { X }  \mathcal { V }$ mapping between two separable Hilbert spaces $\mathcal { X } , \mathcal { y }$ . We endow X with a probability measure ν and, given $X _ { 1 } , \ldots , X _ { m } \sim _ { \mathrm { i . i . d . } }$ ν, consider training data

$$
( X _ { i } , Y _ { i } ) , \ i = 1 , \ldots , m , \quad { \mathrm { w h e r e } } \ Y _ { i } = F ( X _ { i } ) + E _ { i } \in \mathcal { Y }\tag{5.1}
$$

and $E _ { i } \in \mathcal { V }$ represents noise. The task is to approximate $F$ from the data (5.1).

Operator learning is often carried out using DNOs. These are extensions of DNNs that are designed to handle the infinite-dimensional nature of the inputs and outputs. There are many diferent operator learning strategies, such as DeepONets [60], Fourier Neural Operators (FNOs) [55], PCA-Net [48] and many others. We refer to [16, 18, 44, 74] for reviews.

A standard operator learning paradigm, termed encoder-decoder nets, involves approximating $F$ with a DNO of the form

$$
\begin{array} { r } { F \approx \hat { F } : = \hat { \mathcal { D } } _ { \mathcal { Y } } \circ \hat { N } \circ \hat { \mathcal { E } } _ { \mathcal { X } } , } \end{array}\tag{5.2}
$$

where $\hat { \mathcal { E } } _ { \mathcal { X } } : \mathcal { X }  \mathbb { R } ^ { d _ { \mathcal { X } } }$ in an encoder for $\mathcal { X } , \hat { \mathcal { D } } _ { \mathcal { Y } } : \mathbb { R } ^ { d _ { \mathcal { Y } } }  \mathcal { Y }$ is a decoder for $\mathcal { V }$ and $\hat { N } : \mathbb { R } ^ { d _ { \mathcal { X } } }  \mathbb { R } ^ { d _ { \mathcal { Y } } }$ is a neural network. The encoder and decoder are either specified by the problem, learned spearately from data or learned concurrently with $\hat { N }$ . Note that DeepONets and PCA-Net both fall into this category, as do various other approaches. Our main theoretical results consider certain types of encoder-decoder DNOs.

5.1. Assumptions and definitions. We now introduce the key assumptions needed for our theoretical results.

Assumption 5.1. The probability measure ν has mean zero and finite second moments, i.e., $\begin{array} { r } { \int _ { \mathcal { X } } \left\| X \right\| _ { \mathcal { X } } ^ { 2 } \mathrm { d } \mu ( X ) < \infty } \end{array}$ . Further, let $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \dots \geq 0$ and $\{ \phi _ { i } \} _ { i \in \mathbb { N } } \subset \mathcal { X }$ be the eigenvalues and orthonormal basis of eigenvectors of the covariance operator of $\nu .$ Then the real-valued random variables $\xi _ { i } = \langle X , \phi _ { i } \rangle _ { \mathcal { X } } / \sqrt { 3 \lambda _ { i } }$ satisfy $\xi _ { i } \sim _ { \mathrm { i . i . d . } } \mathcal { U } ( [ - 1 , 1 ] )$

Here $\sqrt { 3 }$ is a normalization factor, as $\mathbb { E } \xi _ { i } ^ { 2 } = 1 / 3$ . Assumption 5.1 means that $\nu$ is the law of the X-valued random variable $\begin{array} { r } { X = \sum _ { i = 1 } ^ { \infty } \sqrt { 3 \lambda _ { i } } \xi _ { i } \phi _ { i } } \end{array}$ , where $\xi _ { i } \sim _ { \mathrm { i . i . d } }$ $\mathcal { U } ( [ - 1 , 1 ] )$ ). Such an assumption is quite common in practice. Indeed, it is standard in operator learning literature to generate training data by simulating inputs $X$ via such an expansion [16,44,74]. We slightly deviate from standard practice by assuming the $\xi _ { i }$ are uniformly-distributed random variables, rather than standard normal random variable. In the latter case, ν is a Gaussian process on $\mathcal { X }$ . Dealing with normal random variables presents some technical challenges, which are beyond the scope of this paper. See $\ S 7$ for further discussion.

Assumption 5.2. Let the $\phi _ { i }$ and $\lambda _ { i }$ be as in Assumption 5.1. The encoder $\hat { \mathcal { E } } _ { X }$ is given by $\hat { \mathcal { E } } _ { \mathcal { X } } : X \mapsto ( \langle X , \phi _ { i } \rangle _ { \mathcal { X } } / \sqrt { 3 \lambda _ { i } } ) _ { i = 1 } ^ { d _ { \mathcal { X } } }$ and the decoder $\hat { \mathcal { D } } _ { \mathcal { Y } }$ is linear.

This assumption means that our DNO architecture is a type of PCA-Net [48], at least in terms of its encoder, since $\hat { \mathcal { E } } _ { X }$ involves analyzing $X \in { \mathcal { X } }$ using the first $d _ { \mathcal { X } }$ PCA basis functions. In PCA-Net, one normally first learns the PCA basis empirically from data before learning the DNN $\hat { N }$ . To avoid complications, we assume the exact PCA basis is known. However, we believe it is possible to extend our analysis to the case where an empirical PCA basis is employed by tracking the influence of this error on the overall error bound. This is a topic for future work. In PCA-Net, the decoder $\hat { \mathcal { D } } _ { \mathcal { Y } }$ is also constructed using the empirical PCA basis on $\mathcal { V }$ specified by the pushforward measure $F \sharp \nu$ . However, we do not require this, since we just assume linearity of the map.

Assumption 5.3. The probability measure $\mu$ on $\mathcal { X }$ satisfies the following. Let υ denote the law of the random variable $\boldsymbol { \xi }$ on $\mathbb { R } ^ { \mathbb { N } }$ , defined as $\pmb { \xi } = \left( \langle X , \phi _ { i } \rangle _ { \pmb { \chi } } / \sqrt { 3 \lambda _ { i } } \right) _ { i \in \mathbb { N } } ,$ where $X \sim \mu$ . Then supp $( v ) \subseteq D _ { \omega }$ for some $\omega \geq 1$ , where $D _ { \omega }$ is as in (1.3).

This states that the projection of samples $X \sim \mu$ via the PCA basis $\{ \phi _ { i } \} _ { i \in \mathbb { N } }$ scaled by the PCA eigenvalues $\{ \lambda _ { i } \} _ { i \in \mathbb { N } }$ , are bounded, and the ith such projection $\langle X , \phi _ { i } \rangle _ { \mathcal { X } } / \sqrt { 3 \lambda _ { i } }$ belongs to $[ - \omega _ { i } , \omega _ { i } ]$ almost surely. Note that we do not require the random variables $\left. X , \phi _ { i } \right. _ { \mathcal { X } } / \sqrt { 3 \lambda _ { i } }$ to be uncorrelated, hence this assumption is quite general. We discuss several examples that satisfy it in §5.3 below. In particular, we show that this assumption allows for functions sampled from the test distribution $\mu$ to be less smooth than functions sampled from the training distribution ν – an important practical scenario.

Finally, we now define the class of holomorphic operators we consider in this work.

Definition 5.4. Suppose that Assumption 5.1 holds. We say that $F : \mathcal { X }  \mathcal { Y }$ is $a \ ( \pmb { b } , \varepsilon )$ -holomorphic operator if $F ( X ) \ = \ f ( \langle X , \phi _ { i } \rangle _ { \mathcal { X } } ) _ { i = 1 } ^ { \infty } / \sqrt { 3 \lambda _ { i } } )$ , ∀X ∈ X, and $f : \mathbb { R } ^ { \tilde { \mathbb { N } } }  \mathcal { V }$ is a $( b , \varepsilon )$ -holomorphic Y-valued function. We write $\mathcal { H } ( \boldsymbol { b } , \varepsilon ; \mathcal { V } )$ for the class of Y-valued (b, ε)-holomorphic functions and $\mathcal { H } ( \boldsymbol { b } , \boldsymbol { \varepsilon } ; \mathcal { X } , \mathcal { Y } )$ for the class of $( b , \varepsilon ) \cdot$ holomorphic operators.

Remark 5.5 (Infinite encoders and decoders). It is convenient at this point to define the ‘infinite’ encoder and decoder

$$
\mathcal { E } _ { \mathcal { X } } : X \mapsto ( \langle X , \phi _ { i } \rangle _ { \mathcal { X } } / \sqrt { 3 \lambda _ { i } } ) _ { i \in \mathbb { N } } , \quad \mathcal { D } _ { \mathcal { X } } : \pmb { x } \mapsto \sum _ { i \in \mathbb { N } } \sqrt { 3 \lambda _ { i } } x _ { i } \phi _ { i } .
$$

Since $\{ \phi _ { i } \} _ { i \in \mathbb { N } }$ is an orthonormal basis, we have that $\mathcal { D } _ { \mathcal { X } } \circ \mathcal { E } _ { \mathcal { X } } = \mathcal { I } _ { \mathcal { X } }$ . Notice that Assumption 5.1 is equivalent to $\mathcal { E } _ { \mathcal { X } } \sharp \nu = \varrho .$ , where we recall that $\varrho = \mathcal { U } ( [ - 1 , 1 ] ) ^ { \mathbb { N } }$ and Assumption 5.3 can be written equivalently as supp $( \mathcal { E } _ { \mathcal { X } } \sharp \mu ) \subseteq D _ { \omega }$ . Further, the operator F and function $f$ in Definition 5.4 satisfy $F = f \circ { \mathcal { E } } _ { \mathcal { X } }$ and $f = F \circ D _ { \mathcal { X } }$

5.2. Main result. Our main results consider operator learning using encoderdecoder nets of the form (5.2). Given an encoder $\hat { \mathcal { E } } _ { X }$ and decoder $\hat { \mathcal { D } } _ { \mathcal { Y } }$ we let N be a family of DNNs $N : \mathbb { R } ^ { d _ { \mathcal { X } } } \overset { \cdot } {  } \mathbb { R } ^ { d _ { \mathcal { Y } } }$ and consider the estimator

$$
\hat { F } = \hat { \mathcal { D } } y \circ \hat { N } \circ \hat { \mathcal { E } } _ { \mathcal { X } } , \qquad \mathrm { w h e r e } ~ \hat { N } \in \underset { N \in \mathcal { N } } { \mathrm { a r g m i n } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \big \| Y _ { i } - \hat { \mathcal { D } } y \circ N \circ \hat { \mathcal { E } } _ { \mathcal { X } } ( X _ { i } ) \big \| _ { \mathcal { Y } } ^ { 2 } .\tag{5.3}
$$

Before stating our result we need one further piece of notation. Recall from Assumption 5.2 that $\hat { \mathcal { D } } _ { \mathcal { Y } } : \mathbb { R } ^ { d _ { \mathcal { Y } } }  \mathcal { Y }$ is linear, therefore its range $\hat { \mathcal { Y } } : = \hat { \mathcal { D } } _ { \mathcal { Y } } ( \mathbb { R } ^ { d _ { \mathcal { Y } } } )$ denotes an at most d<sub>Y</sub>-dimensional subspace. We now write $\hat { B } _ { y } : \mathcal { V } \to \hat { \mathcal { V } }$ for the orthogonal projection onto Y<sup>ˆ</sup>.

Theorem 5.6. Let $0 < \epsilon < 1$ and $m \geq { \bar { m } }$ , where $\bar { m } = \bar { m } ( \epsilon )$ depends on ϵ only. Suppose that Assumptions 5.1-5.3 hold with $\omega >$ 1 satisfying $\omega _ { \mathrm { m i n } } : = \mathrm { m i n } \{ \omega _ { i } \} > 1$ Then there exists a DNN model class N with $n _ { 0 } = d _ { \mathcal { X } }$ and $n _ { M + 1 } = d _ { \mathcal { Y } }$ and depending on m, ω and ϵ only with the following property. Let $F \in \mathcal H ( b , \varepsilon ; \mathcal { X } , \mathcal { Y } )$ , where $\mathbf { \delta } _ { b } \geq \mathbf { 0 }$ and $\varepsilon > 0$ satisfy (4.2) for some $0 < p < 1$ , draw $X _ { 1 } , \dots , X _ { m } \sim _ { \mathrm { i . i . d } } $ ν and let $\hat { F }$ be as in (5.3) with noisy training data (5.1). Then, with probability at least $1 - \epsilon , \hat { F }$ satisfies

$$
\begin{array} { r l } & { \displaystyle \| F - \hat { F } \| _ { L _ { \mu } ^ { \infty } ( \mathcal { X } ; \mathcal { Y } ) } \lesssim \frac { C ( b , \varepsilon , \omega , p ) } { \sqrt { \epsilon } } \left( \frac { m } { L } \right) ^ { 1 - 1 / p } + \sqrt { \frac { m } { \epsilon L } } \| F - \hat { B } _ { \mathcal { Y } } \circ F \| _ { L _ { \nu } ^ { 2 } ( \mathcal { X } ; \mathcal { Y } ) } } \\ & { \quad \quad \quad \quad + \frac { \| E \| _ { 2 ; \mathcal { Y } } } { \sqrt { L } } + 2 ^ { - m } , } \end{array}
$$

provided $d _ { \mathcal { X } } \ge c m / L \ f o r$ some universal constant $^ { c , }$ where $L = L ( m , \epsilon ) : = \log ^ { 4 } ( m ) +$ $\log ( 1 / \epsilon )$ and $\pmb { { \cal E } } = ( E _ { i } ) _ { i = 1 } ^ { m }$ . Further, the class $\mathcal { N }$ satisfies

$$
\mathrm { w i d t h } ( { \cal N } ) \lesssim \frac { m } { \log ( \omega _ { \mathrm { m i n } } ) } , \qquad \operatorname * { d e p t h } ( { \cal N } ) \lesssim \log \left( \frac { \log ( m ) } { \log ( \omega _ { \mathrm { m i n } } ) } \right) .
$$

Here and elsewhere $\| \cdot \| _ { L _ { \mu } ^ { p } ( \mathcal { X } ; \mathcal { Y } ) }$ denotes the L<sup>p</sup>-Bochner norm (see §SM1.1). This result provides an OOD generalization guarantee for operator learning. It shows that the generalization error for learning holomorphic operator decays with algebraic rates of convergence in terms of m depending on the support of the measure $\mu ,$ as described by Assumption 5.3. We discuss examples of this condition in the next subsection.

As with previous results, the DNNs are not excessively wide or deep. Similar to Theorem 4.4, while this result considers tanh DNNs, other activations could readily be considered. Distinct from Theorem 4.4, however, this result describes learning classes of operators taking values in an arbitrary separable Hilbert space $\mathcal { V }$ . For this reason, the error bound contains an additional term that acounts for the decoding error on $y { : }$ namely,

$$
\| F - \hat { B } _ { \mathcal { Y } } \circ F \| _ { L _ { \nu } ^ { 2 } ( \mathcal { X } ; \mathcal { Y } ) } = \sqrt { \int _ { \mathcal { X } } \left\| F ( X ) - \hat { B } _ { \mathcal { Y } } \circ F ( X ) \right\| _ { \mathcal { Y } } ^ { 2 } \mathrm { d } \nu ( X ) } \equiv \| \mathcal { Z } _ { \mathcal { Y } } - \hat { B } _ { \mathcal { Y } } \| _ { L _ { F \sharp \nu } ^ { 2 } ( \mathcal { Y } ; \mathcal { Y } ) } .
$$

Here $\mathcal { I } _ { \mathcal { V } } : \mathcal { V }  \mathcal { V }$ is the identity map. Such an expression is quite standard in the operator learning literature. Overall, this term implies that the decoder error depends on how well $\hat { B } _ { y }$ approximates the identity map. Concrete estimates for this term have been established in various settings, such as PCA-Nets and DeepONets [48, 49].

5.3. Examples and further discussion. We now discuss Assumption 5.3 in more detail.

Example 5.7 (Rougher test distributions). Suppose that $\mu$ has the same PCA eigenbasis as $\nu ,$ with PCA eigenvalues $\{ \lambda _ { i } ^ { \prime } \} _ { i \in \mathbb { N } }$ satisfying $\lambda _ { i } ^ { \prime } \geq \lambda _ { i } , \forall i \in \mathbb { N }$ . In this case, $\mu$ is the law of the random variable $\begin{array} { r } { X = \sum _ { i = 1 } ^ { \infty } \xi _ { i } \sqrt { 3 \lambda _ { i } ^ { \prime } } \phi _ { i } } \end{array}$ , where $\xi _ { i } \sim _ { \mathrm { i . i . d . } } \mathcal { U } ( [ - 1 , 1 ] )$ Note that, $i f \left\{ \phi _ { i } \right\} _ { i \in \mathbb { N } }$ is an orthonormal basis of smooth functions, such as the Fourier basis, this means that functions sampled from the test distribution µ can be less smooth than functions sampled from the training distribution ν. In this case, Assumption 5.3 holds if and only if

$$
\lambda _ { i } ^ { \prime } \leq \omega _ { i } ^ { 2 } \lambda _ { i } , \quad \forall i \in \mathbb { N } .
$$

meaning that the eigenvalues $\lambda _ { i } ^ { \prime }$ may decay as $i  \infty$ at a slower rate than the $\lambda _ { i }$ , depending on the growth rate of the $\omega _ { i }$ . Indeed, after choosing $\omega$ minimally so that the above inequality holds, we obtain the OOD rate $\mathcal { O } ( ( m / \bar { L } ) ^ { 1 - 1 / p } )$ provided $F \in \mathcal H ( b , \varepsilon ; \mathcal { X } , \mathcal { Y } )$ for b, ε satisfying

$$
b \odot r _ { p } \in \ell _ { \mathsf { M } } ^ { p } ( \mathbb { N } ) , \quad \| b \odot ( r _ { p } - \mathbf { 1 } ) \| _ { 1 } < \varepsilon , \qquad w h e r e ~ r _ { p } = \left( \zeta \left( \sqrt { \lambda _ { i } ^ { \prime } / \lambda _ { i } } \right) ^ { 2 / p - 1 } \right) _ { i \in \mathbb { N } } .\tag{5.4}
$$

Concretely, let $\lambda _ { i } = i ^ { - v }$ and $\lambda _ { i } ^ { \prime } = i ^ { - u } ~ f o r ~ v \ge u > 1$ and suppose, as in Example $4 . 6 ,$ that $b _ { i } = c i ^ { - a }$ for some $c > 0$ . Then, it is a short argument to show that algebraic convergence occurs – $i . e . , \ ( 5 . 4 )$ holds for some $p \in ( 0 , 1 )$ – whenever $\textstyle a > 1 + { \frac { v - u } { 2 } }$ and c is suficiently small. Thus, the OOD error decays algebraically even when $\mu$ is much rougher than $\nu \ ( i . e . , u \ll v )$ , provided F has suficient regularity.

In the previous example, the entries $\xi _ { i }$ of the random variable ξ defined in Assumption 5.3 were uncorrelated, since ν and $\mu$ difered only in their PCA eigenvalues. However, Assumption 5.3 allows for correlated distributions, thus allowing ν and $\mu$ to be very diferent in nature. We now explore several such examples

Example 5.8 (Arbitrarily rough test distributions). Suppose that $\| X \| _ { \mathcal { X } } \leq c$ for $X \sim \mu \ a . s .$ .. Note that such a distribution can be arbitrarily rough. Assumption 5.3 holds, provided $\omega _ { i } \geq c / \sqrt { 3 \lambda _ { i } } , \forall i \in \mathbb { N }$ . In this case, Theorem 5.6 shows that one can generalize to arbitrarily rough test distributions given suficient smoothness: namely, $( b , \varepsilon )$ -holomorphy, where b satisfies (4.2) for some ω whose entries $\omega _ { i }$ grow proportional to $1 / \sqrt { \lambda _ { i } }$ . Thus, and perhaps unsurprisingly, faster decay of the eigenvalues $\lambda _ { i } ~ ( i . e . ,$ , a smoother training distribution ν) translates into higher smoothness of the operator in order to ensure convergence for an arbitrarily rough test distribution $\mu .$

For a concrete example, let $\mathcal { X } = L ^ { 2 } ( [ 0 , 1 ] )$ and $\phi _ { i } ( x ) = \sqrt { 2 } \sin ( i \pi x ) , \forall i \in \mathbb { N }$ , be the sine basis. Let ν be the law of the random variable $\begin{array} { r } { X = \sqrt { 2 } \sum _ { i = 1 } ^ { \infty } \xi _ { i } i ^ { - v / 2 } \sin ( i \pi \cdot ) } \end{array}$ ， where $\xi _ { i } \sim _ { \mathrm { i . i . d . } } \ u ( [ - 1 , 1 ] )$ , for some $v > 1$ . Now let $d > 1$ and $\mu$ be the law of the random variable $\begin{array} { r } { X = \sum _ { i = 0 } ^ { d - 1 } ( 2 + \xi _ { i } ) \mathbb { I } _ { [ i / d , ( i + 1 ) / d ) } ( \cdot ) } \end{array}$ , where $\xi _ { i } \sim _ { \mathrm { i . i . d . } } \ \mathcal { U } ( [ - 1 , 1 ] )$ Functions drawn from ν belong almost surely to $H ^ { s } ( [ 0 , 1 ] )$ for $s < ( v - 1 ) / 2$ , while functions drawn from $\mu$ are pieceiwse constant and only belong to $H ^ { s } ( [ 0 , 1 ] )$ almost surely for $s < 1 / 2$ . Since $\| X \| _ { \mathcal { X } } \leq 3$ for $X \sim \mu$ a.s. we may choose $\bar { \omega } _ { i } = \bar { 3 } i ^ { v / 2 }$ in order for Assumption 5.3 to hold. Such choice is independent of d. However, a direct calculation gives that $| \langle X , \phi _ { i } \rangle _ { \mathcal { X } } | \leq 6 \sqrt { 2 } d / i , \forall i \in \mathbb { N }$ , almost surely for $X \sim \mu$ , meaning that one can also choose $\omega _ { i } = \operatorname* { m a x } \{ 1 , 6 \sqrt { 2 } d i ^ { v / 2 - 1 } \}$ . Therefore, if d is not large, a more slowly-growing choice $o f \omega _ { i } \textrm { -- } w h i c h$ equates to a less stringent smoothness requirement on $F \mathrm { ~ - ~ } i s$ also possible.

6. Numerical experiments. We now present a series of numerical experiments. These experiments consider polynomial, DNN and DNO estimators, with full implementation details given in Appendix SM2. The intention of these experiments is to illustrate the main messages of the paper: namely, OOD generalization errors decay at algebraic rates depending on the support of the test distribution. But they also elucidate a number of gaps between the theory developed in this work and practice. This motivates future work, as we discuss further in $\ S 7$

Remark 6.1 (Unknown supports). The estimators we consider in our experiments are all independent of $\omega ,$ the parameter defining the hyperrectangle $D _ { \omega }$ in which the support of the test distribution $\mu$ is contained. This is a key diference with our theory, since the estimators of Theorems 4.1, 4.4 and 5.6 all depend on $\omega .$ . An important takeaway from our experiments is that polynomial and DNN estimators possess even stronger OOD performance than the theory suggests, as they achieve algebraic rates of convergence without any knowledge of the test distribution $\mu .$ . As we discuss further in $\ S 7$ , understanding why this is the case is an open problem.

6.1. Polynomial estimators for multivariate function approximation. We first consider multivariate function approximation. Our aim is to conduct a precise study of the efect of changing the support of the test distribution on the convergence rate and to compare the empirically-observed rates to the theoretical rates. Therefore, in order to avoid efects implicit to training DNNs – such as architecture design, initializations and optimization solvers – in this subsection we consider polynomial estimators. For the reasons discussed in Remark 6.2, we do not implement exact the estimator of Theorem 4.1. Instead, we consider an eficient Adaptive Least Squares (ALS) estimator based on [61]. This estimator is described in §SM2.1.

Remark 6.2 (Diferences between Theorem 4.1 and the experiments). There are several reasons why we do not implement the estimator of Theorem 4.1. First, as discussed in Remark 4.3 it is generally impracticable. While it is possible to design an implementable estimator using weighted ℓ<sup>1</sup>-minimization, such estimators can be memory-intensive and slow to compute [3], thus limiting ones ability to perform convergence studies involving large sample sizes. In comparison, the ALS estimator is relatively inexpensive to compute, yet it lacks theoretical guarantees, even in the indistribution setting [61]. However, the experiments in this section suggest it possesses excellent in-distribution and OOD generalization properties.

To conduct a precise study, we consider the family of functions

$$
f : [ - 1 , 1 ] ^ { d } \to \mathbb { R } , \ x = ( x _ { i } ) _ { i = 1 } ^ { d } \mapsto \prod _ { i = 1 } ^ { d } \frac { \sqrt { 2 \delta _ { i } + \delta _ { i } ^ { 2 } } } { x _ { i } + 1 + \delta _ { i } } ,\tag{6.1}
$$

for positive parameters $\delta _ { i } > 0$ (the factor in the numerator ensures this function has unit norm, and therefore avoids scale efects for large d due to the d-fold product) [2]. These functions are holomorphic in $[ - 1 , 1 ] ^ { d }$ with singularities at any x for which $x _ { i } = - 1 - \delta _ { i }$ for some i. Therefore they are holomorphic in $\mathcal { E } _ { \rho _ { 1 } } \times \cdots \times \mathcal { E } _ { \rho _ { d } }$ for any $\rho _ { i } \geq 1$ satisfying

$$
\rho _ { i } < \zeta ( 1 + \delta _ { i } ) \quad \forall i \in [ d ] ,\tag{6.2}
$$

where we recall $\zeta$ from §1.2. Because of Remark 2.3, the function $( 6 . 1 ) \textrm { -- } \mathrm { o r } .$ , more precisely, its extension to $[ - 1 , 1 ] ^ { \mathbb { N } } - { \mathrm { i s ~ } } ( \pmb { b } , \varepsilon )$ -holomorphic for any $\pmb { b } = ( b _ { i } ) _ { i = 1 } ^ { \infty }$ satisfying

$$
b _ { i } > \varepsilon / \delta _ { i } , \forall i \in [ d ] , \qquad b _ { i } = 0 , \forall i > d .\tag{6.3}
$$

In our experiments, we consider $\delta _ { i } = 2 i ^ { 2 }$ . For training and and test distributions, we consider $\varrho = \mathcal { U } ( [ - 1 , 1 ] ^ { d } )$ and $\mu = \mathcal { U } \left( \otimes _ { i = 1 } ^ { d } \left[ - \omega _ { i } , \omega _ { i } \right] \right)$ , respectively.

Following Example 4.5, in Fig. 2 we first consider constant side lengths $\omega _ { 1 } = \cdots =$ $\omega _ { d } = \omega \geq 1$ . The largest value $\omega = 1 . 8$ is close to the largest possible hypercube in which $f$ is holomorphic. Indeed, min $; \delta _ { i } = \delta _ { 1 } = 1$ and hence (6.2) implies that $f$ is holomorphic in any hypercube $[ - \omega , \omega ] ^ { d }$ with $\omega < 2$ . Fig. 2 shows that the OOD error decays for all values of ω considered. Recalling Remark 2.5, we notice that in lower dimension $d = 8$ , the rate of convergence appears slightly faster than algebraic, while when d increases to $d = 3 2$ we witness algebraic convergence.

In Fig. 2, we also compare the empirically-obtained algebraic rate of convergence q<sub>fit</sub> with the rate predicted by our theory $q _ { \mathrm { t h y } }$ . Recall that our theory predicts a rate $\bar { ( \boldsymbol { m } / L ) } ^ { 1 - 1 / p }$ whenever (4.2) holds for some $0 < p < 1$ . For a given value of ω, recalling (6.3) and the fact that $\dot { \delta } _ { i } = 2 i ^ { 2 }$ , we determine that

$$
q _ { \mathrm { t h y } } = \frac { 1 } { 2 } \left[ \log \left( 1 + 2 \Big / \sum _ { i = 1 } ^ { d } i ^ { - 2 } \right) \Big / \log ( \zeta ( \omega ) ) - 1 \right] ,\tag{6.4}
$$

meaning that our theory predicts a rate that is $m ^ { - q _ { \mathrm { t h y } } + \delta }$ for any $\delta > 0$

As Fig. 2 shows, the estimator consistently outperforms the theoretical convergence rate, except in the case $\omega = 1$ , which corresponds to the in-distribution error. Further, after manipulating (6.4), we see that $q _ { \mathrm { t h y } } \leq 0$ , meaning our theory predicts no convergence, whenever $\omega \geq 1 . 3 4 3$ . Yet, the estimator still converges algebraically for $\omega = 1 . 6$ and $\omega = 1 . 8$ . This suggests the main condition (4.2) may not be sharp – improving it is an open problem.

Next, we consider varying side lengths. Following Example 4.6, we set $\omega _ { i } = i ^ { b }$ for several diferent values of b. Similar to the previous example, we witness algebraic convergence, or faster in the $d = 8$ case. We also compare the rate against our theory. In this case, the theoretical rate is given by

$$
q _ { \mathrm { t h y } } = 1 / p _ { \mathrm { t h y } } - 1 , \quad { \mathrm { w h e r e ~ } } p _ { \mathrm { t h y ~ s o l v e s } } \sum _ { i = 1 } ^ { d } \left( \zeta ( i ^ { b } ) ^ { 2 / p - 1 } - 1 \right) i ^ { - 2 } = 2 .\tag{6.5}
$$

![](images/95613b400985824d0c1dd59cff33da951ad1b1df76e8dbb47bd4dfa184154ba0.jpg)

![](images/f31bfc5fcfaea9a239174ed2f2ac22873f70ee4939aa38b267e5ce035d8cc4df.jpg)

<table><tr><td>ω</td><td>qfit</td><td>Qthy</td></tr><tr><td>1.0</td><td>1.004</td><td>1.000</td></tr><tr><td>1.1</td><td>0.960</td><td>0.409</td></tr><tr><td>1.2</td><td>0.912</td><td>0.148</td></tr><tr><td>1.3</td><td>0.854</td><td>0.033</td></tr><tr><td>1.6</td><td>0.689</td><td></td></tr><tr><td>1.8</td><td>0.564</td><td></td></tr></table>

Fig. 2: Error versus number of training samples m for function approximation using polynomial estimators computed by the ALS algorithm. The function is given by (6.1) with $2 \delta _ { i } = i ^ { 2 }$ , ∀i ∈ [d] and $d = 8$ (left) or $d = 3 2$ (middle). The training and test distributions are $\varrho = \mathcal { U } ( [ - 1 , 1 ] ^ { d } )$ and $\mu = \mathcal { U } ( \otimes _ { i = 1 } ^ { d } \left[ - \omega _ { i } , \omega _ { i } \right] )$ , where $\omega _ { 1 } = \cdot \cdot \cdot = \omega _ { d } = \omega$ . The value $q _ { \mathrm { t h y } }$ is given by (6.4) in the case $d = 3 2$ . A dashed line means that $q _ { \mathrm { t h y } } \leq 0 ,$ , i.e., no convergence. The values $q _ { \mathrm { f i t } }$ were computed by empirical fitting in the case $d = 3 2$

![](images/b3d6092bb2161ce995fc95a4f5107aa441d93a975e3c3a50cb0a5d17655de6ac.jpg)

![](images/580d4912f4738f428bf1ab993def6b1b135fa2029ba24a8e0a41a801ade31545.jpg)

<table><tr><td>b</td><td>qfit</td><td>Qthy</td></tr><tr><td>0.0</td><td>1.026</td><td>1.000</td></tr><tr><td>0.1</td><td>0.951</td><td>0.855</td></tr><tr><td>0.2</td><td>0.864</td><td>0.430</td></tr><tr><td>0.3</td><td>0.782</td><td>0.237</td></tr><tr><td>0.4</td><td>0.707</td><td>0.120</td></tr><tr><td>0.5</td><td>0.660</td><td>0.039</td></tr><tr><td>0.6</td><td>0.581</td><td></td></tr><tr><td>0.8</td><td>0.488</td><td></td></tr></table>

Fig. 3: The same as Fig. 2, except with varying side lengths $\omega _ { i } = i ^ { b }$ and the value $q _ { \mathrm { t h y } }$ given by (6.5) in the case $d = 3 2$ once more.

Once more, the empirical convergence rate is better than our theory predicts. In particular, the estimator converges for $b = 0 . 6$ and $b = 0 . 8$ . This is outside the realm of our theory, which only asserts convergence for $b < 0 . 5 6 2$ . Indeed, (6.5) implies that $q _ { \mathrm { t h y } } \leq 0$ whenever $b \geq 0 . 5 6 2$ . This once again points towards a potential lack of sharpness in the condition (4.2) and an open problem for future work.

6.2. DNN estimators for multivariate function recovery. We now consider DNN estimators. Rather than examine precise algebraic rates, our goal is to demonstrate the OOD performance for a series of diferent functions motivated by applications, in scenarios where the support of the test distribution varies. As in the previous section, we do not implement the estimator described in our theoretical result (Theorem 4.4). Instead, we opt for a fully-connected, feedforward architecture that is independent of $\omega$ (recall Remark 6.1). Details of the architecture and training procedure are given in §SM2.2.

We consider several standard test functions from the Virtual Library of Simulation Experiments [75], specifically, the Physical Models section of the Emulation/Prediction Test Problems test set. This is a standard library for high-dimensional approximation and integration tasks, where each function represents a model for a certain physical process. See [75] for details. To align with the rest of the paper, we rescale the input variables to $[ - 1 , 1 ] ^ { d }$ and consider $\varrho = \mathcal { U } ( [ - 1 , 1 ] ^ { d } )$ as the training distribution.

In Fig. 1, we consider the wing weight function. This is a $d \ = \ 1 0$ variate function, which, after rescaling to $[ - 1 , 1 ] ^ { 1 0 }$ is holomorphic inside the hyperrectangle $D _ { \omega } = \otimes _ { i = 1 } ^ { 1 0 } \bigl [ - \omega _ { i } , \omega _ { i } \bigr ]$ for any $\omega = ( \omega _ { i } ) _ { i = 1 } ^ { 1 0 }$ satisfying $\omega < \omega _ { \mathrm { m a x } }$ , where $\omega _ { \mathrm { m a x } } =$ $( 7 , 6 . 5 , 4 , 9 , 2 . 1 0 , 3 , 2 . 6 , 2 . 4 3 , 5 . 2 5 , + \infty )$ . In Fig. 1 we consider $\mu = \mathcal { U } ( D _ { \omega } )$ , where $\omega _ { i } =$ $1 + \theta ( \omega _ { i , \operatorname* { m a x } } - 1 ) , \forall i \in [ 1 0 ]$ , with the value $\omega _ { \mathrm { m a x , 1 0 } } = 1 0$ . In other words, we study extrapolation to a domain whose side lengths are a constant fraction of the maximum possible extrapolation domain. As expected, in all cases we see algebraic convergence with the rate depending on the size of θ. Note that we plot the squared $L ^ { 2 } .$ -error in this experiment, in order to make a comparison with the distributional shift bound (1.2). As we see from the table, the Wasserstein distance $W _ { 2 } ( \varrho , \mu )$ is significantly larger than this error, thus demonstrating the pessimistic nature of such bounds.

![](images/dae04820a1bc99d2338053f41e3c8b607ee4952bc365b5e83050736130eb26c2.jpg)  
Fig. 4: Error versus number of training samples m for approximating the circuit (left) and piston (right) functions via tanh DNNs with $L = 1 5$ layers and width $W = 1 5 0 .$

In Fig. 4 we consider two other functions from the same library, the circuit and piston functions. These are $d = 6$ and $d = 7$ variate functions, respectively, that are holomorphic inside the hyperrectangle $D _ { \omega } = \otimes _ { i = 1 } ^ { 6 } [ - \omega _ { i } , \omega _ { i } ]$ for any $\boldsymbol { \omega } = ( \omega _ { i } ) _ { i = 1 } ^ { 6 }$ satisfying $\omega = ( \omega _ { 1 } , \textrm { , . . . , } \omega _ { 6 } ) < \omega _ { \mathrm { m a x } } : = ( 2 , 2 . 1 1 , 1 . 4 0 , 2 . 8 5 , 2 0 . 4 2 7 , 1 . 4 0 )$ and $\omega = ( \omega _ { 1 } , \ldots , \omega _ { 7 } ) <$ $\omega _ { \mathrm { m a x } } : = ( 3 , 1 . 6 7 , 1 . 5 , 1 . 5 , 1 0 . 9 7 , 6 7 , 3 5 )$ , respectively. We consider $\varrho = \mathcal { U } ( [ - 1 , 1 ] ^ { d } )$ and $\mu = \mathcal { U } ( [ - \omega , \omega ] ^ { d } )$ , where $\omega \geq 1$ is a parameter we vary. For both functions, we witness algebraic convergence for all values of ω. It is noticeable that the algebraic rate decreases slightly as ω increases. For the circuit function, the algebraic power decreases from 0.4 in the $\omega = 1$ (i.e., in distribution) case to 0.19 in the $\omega = 1 . 3$ case. Similarly, for the piston function it decreases from 0.44 when $\omega = 1$ to 0.21 when $\omega = 1 . 4$

Remark 6.3 (Diferences between Theorem 4.4 and numerical experiments). The primary diference is the choice of DNN family N. As discussed in 4.2, Theorem 4.4 uses a specific handcrafted family of DNNs, where only certain values for the internal weights and biases are allowed. Conversely, in our numerical experiments, we consider fully-connected, feedforward DNNs where all weights and biases are trained and can take arbitrary real values. In particular, the resulting estimator is independent of ω, whereas the DNN family, and consequently the DNN estimator, of Theorem 4.4 depends on ω (recall Remark 6.1).

6.3. DNO estimators for PDE operators. Finally, we consider operator learning. As in the previous subsections, we do not strive to implement the DNO estimators described in Theorem 5.6. Instead, due to their wide popularity and robust performance, we consider FNOs. Details of the architecture and training procedure

![](images/b0b8cb90bab576e5bbf98ee347696de0691a0557dca8e85f8a220bc4cb47cc35.jpg)

![](images/32121ae318e4baa4f5fb1d5bd3551098ae71eef41a955c38d0c5374814e74301.jpg)

![](images/0012a8fe53f4b2ecbede256ccceb61d1c230f1a50e8f5f4612ac1d42495e6b8d.jpg)  
Fig. 5: Error versus number of training samples m for the Darcy flow problem with $\nu = \mu _ { 3 , 2 } ^ { \mathrm { L N } }$ and $\mu = \mu _ { 3 , \alpha } ^ { \mathrm { L N } }$ (left), $\mu = \mu _ { 3 , \alpha } ^ { \mathrm { P C } }$ (middle); $\nu = \bar { \mu } _ { 3 , 1 } ^ { L N }$ and $\mu = \mu ^ { \mathrm { c o o k i e s } }$ (right).

are given in §SM2.3.

In our first example, we consider the Darcy flow PDE

$$
- \nabla \cdot ( a ( x ) \nabla u ( x ) ) = f ( x ) , \ x \in D , \qquad u ( x ) = 0 , \ x \in \partial D .\tag{6.6}
$$

Here $D = ( 0 , 1 ) ^ { 2 } , u ( x )$ is the pressure field, $f ( x )$ is the forcing function and $a ( x )$ is the input permeability field. We take $f ( x ) = 1$ and consider the operator $F$ mapping from the difusion coeficient $a ( x )$ to the solution $u ( x ) , { \mathrm { i . e . , } } F : a \mapsto u .$

In Fig. 5, we follow standard practice and consider log-normal random fields, with corresponding probability measures

$$
\mu _ { \tau , \alpha } ^ { \mathrm { L N } } = \psi \# \mathcal { N } \left( 0 , ( - \Delta + \tau ^ { 2 } I ) ^ { - \alpha } \right) , \qquad \mathrm { w h e r e } ~ \psi ( x ) = \exp ( x ) .\tag{6.7}
$$

Here $\Delta$ is the Laplacian, which we equip with homogeneous Neumann boundary conditions. In Fig. 5 (left) we consider $\bar { \nu } \ : = \ : \mu _ { 3 , 2 } ^ { \mathrm { L N } }$ for training and for testing we consider $\mu = \mu _ { 3 , \alpha } ^ { \mathrm { L N } }$ for various $\alpha < 2$ . This corresponds to the setting of Example $5 . 7 ,$ since the eigenvalues of the covariance operator of N $\left( 0 , ( - \Delta + \tau ^ { 2 } I ) ^ { - \alpha } \right)$ are $( \pi ^ { 2 } ( k _ { 1 } ^ { 2 } +$ $k _ { 2 } ^ { 2 } ) + \tau ^ { 2 } ) ^ { - \alpha }$ for $k _ { 1 } , k _ { 2 } \ \in \ \mathbb { N } _ { 0 }$ , which means that the test functions are rougher for smaller α. Fig. 5 illustrates the same qualitative behaviour predicted by our theory: the OOD generalization error decays at an algebraically in $m _ { : }$ , at a rate that decreases as $\alpha  1 ^ { + }$ . Next we test the conclusions of Example 5.8. We once more consider $\nu = \mu _ { \tau , \alpha } ^ { \mathrm { L N } }$ , while for testing we use distributions of piecewise constant functions. We consider two examples. The first is a standard distribution in operator learning [55], denoted $\mu _ { \tau , \alpha } ^ { \mathrm { P C } }$ , which takes the same form as (6.7), except with $\psi ( x )$ replaced by a piecewise constant function $\phi$ taking value $\phi ( x ) = 2$ when $x > 0$ and $\phi ( x ) = 0 . 5 $ otherwise. The second is the ‘cookies’ distribution, denoted $\mu ~ = ~ \mu ^ { \mathrm { c o o k i e s } }$ , which is common in parametric PDEs [8, 20]. See (SM2.2). In both cases, we see the behaviour expected: namely, algebraic convergence (albeit at a slow rate) even though realizations from the test distribution are very diferent from those seen in training.

In our second example, we consider the incompressible Navier–Stokes equations in vorticity form on the torus $\mathbb { T } ^ { 2 }$

$$
\begin{array} { r l r } { \partial _ { t } w ( x , t ) + u ( x , t ) \cdot \nabla w ( x , t ) = \nu \Delta w ( x , t ) + f ( x ) , } & { x \in  { \mathbb { T } } ^ { 2 } , \quad t \in ( 0 , T ] , } \\ { \nabla \cdot u ( x , t ) = 0 , } & { x \in  { \mathbb { T } } ^ { 2 } , \quad t \in [ 0 , T ] , } \\ { w ( x , 0 ) = w _ { 0 } ( x ) , } & { x \in  { \mathbb { T } } ^ { 2 } . } \end{array}\tag{6.8}
$$

Here, u is the velocity field, $w = \nabla \times u$ is the vorticity, $w _ { 0 }$ is the initial vorticity, $\nu > 0$ is the viscosity and $f$ is a forcing function. We learn the solution operator

$F : w _ { 0 } \longmapsto w | _ { \mathbb { T } ^ { 2 } \times ( 0 , T ] }$ . Our setup is a standard one in the operator learning literature [55]. For training and test distributions, we use $\mu _ { \tau , \alpha } = \mathcal { N } ( 0 , \sigma ( - \Delta + \tau ^ { 2 } I ) ^ { - \alpha } )$ , where $\Delta$ is equipped with periodic boundary conditions. Throughout, we use parameters $\sigma = 7 ^ { 3 / 2 } , \tau = 7 . 5$ $\nu = 1 0 ^ { - 3 }$ (equivalently R $\mathrm { ~ e ~ } = 1 0 ^ { 3 } )$ $T \ = \ 5 0$ and the forcing ， $f ( x ) = 0 . 1 ( \sin ( 2 \pi ( x _ { 1 } + x _ { 2 } ) ) + \cos ( 2 \pi ( x _ { 1 } + x _ { 2 } ) ) )$ . For training, we use $\alpha = 2 . 5$ while in testing we use various values $\alpha < 2 . 5$ , thus producing rougher test distributions.

Fig. 1 gives our results. Once more, we see algebraic convergence for all values of $\alpha ,$ , with the rate decreasing as $\alpha  1 ^ { + }$ . In this figure, we plot the squared $L ^ { 2 } \mathrm { - e r r o r }$ , to make a direct comparison with the bounds (1.2). Notably, the errors are significantly smaller than the corresponding Wasserstein distances $W _ { 2 } ( \nu , \mu )$ , thus demonstrating the inadequacy of distributional shift theory in explaining the OOD generalization performance of operator learning models, which is the main thesis of this work.

Remark 6.4 (Diferences between Theorem 5.6 and the experiments). As in Remark 6.3, the architectures used in our experiments (i.e., FNOs) difer from those used in Theorem 5.6, which are handcrafted PCA-Net-type architectures. Also, we use Gaussian or lognormal random fields for training in our numerical experiments, while Assumption 5.1 means that in our theory the training distribution has a Karhunen– Lo\`eve expansion defined by i.i.d. $\mathcal { U } ( [ - 1 , 1 ] )$ random variables, rather than $\mathcal { N } ( 0 , 1 )$ random variables. As we discuss below, extending our theory to deal with the latter – and consequently Gaussian random fields – is an open problem.

7. Conclusion and future work. OOD generalization is one of the biggest challenges in SciML. While standard theory shows robustness of estimators to small distributional shifts, it has often been observed that estimators generalize much further beyond their training distribution. This paper provides a theoretical analysis of this phenomenon. Working with classes of holomorphic functions and operators, we derived convergence rates for OOD generalization that depend on the specific regularity and the support of the test distribution. These show successful generalization well beyond small distributional shifts and genuine extrapolation behaviour. Our results are quantifiable, distribution agnostic (i.e., depending only on the support) and, in certain cases, the rates are optimal. We also show that stable extrapolation is possible while retaining algebraic rates, which we refer to as a blessing of high dimensionality. Finally, in the context of operator learning, our results show it is possible to generalize from smooth training distributions to arbitrarily rough test distributions. Numerical results on function and operator learning tasks support our main conclusions. There are various avenues for future work, several of which we now describe.

From adaptation to generalization. As noted in Remark 6.1, our theoretical estimators require knowledge of $\omega _ { : }$ i.e., a support estimate for the test distribution $\mu .$ Thus, our results are of domain adaptation form. In domain adaptation [10, 11] one uses information about $\mu$ to construct ML models that generalize beyond their training set. Interestingly, our results require only very weak knowledge of $\mu \colon$ namely, a support estimate. However, the estimators used in our numerical results are independent of $\omega$ (see Remark 6.1), thus they pertain to the domain generalization setting. An important question is whether one can construct estimators independent of $\omega$ that yield the same generalization bounds.. This would provide a theoretical explanation for the domain generalization behaviour observed in practice.

Weaker conditions and sharper rates. Our theoretical results provide a suficient condition, (4.2), linking holomorphic regularity to the admissible extrapolation domain and the convergence rate. The experiments in §6.1 motivate investigating whether this condition can be weakened. They show convergence for domains larger than those ensured by (4.2), as well as faster rates. Note, however, that these observations concern particular functions, and do not determine worst-case convergence rates of the specific holomorphic function classes. As discussed in Remark 4.7, the theoretical rates are optimal up to logarithmic factors for certain domains with constant side lengths. Establishing sharp rates for growing side lengths, as in Example 4.6, remains open. Resolving these questions requires lower bounds that account for both the extrapolation domain and the required stability to perturbations.

Gaussian distributions. Our function approximation results consider the training distribution $\varrho = \mathcal { U } ( [ - 1 , 1 ] ^ { \mathbb { N } } )$ and arbitrary test distributions supported in the hyperrectangles $D _ { \omega }$ . A natural extension is to tensor-product Jacobi distributions on $[ - 1 , 1 ] ^ { \mathbb { N } }$ , using orthonormal Jacobi polynomials in place of Legendre polynomials. However, distributions with unbounded support present significantly more dificulties. This impacts the operator learning case as well, as in Assumption 5.1 we impose that the random variables $\xi _ { i } \sim \mathcal { U } ( [ - 1 , 1 ] )$ . In particular, our analysis does not apply to the case where ν is a Gaussian measure on $x ,$ as in that case one would have $\xi _ { i } \sim \mathcal { N } ( 0 , 1 )$ . Gaussian measures are ubiquitous in operator learning and, for these reasons, we also used them in our numerical experiments. An important topic for future work is to generalize our analysis to this case. We note, however, that this limitation is not unique to our work – existing statistical learning theory guarantees for operators typically assume boundedness of the training and test distributions [57,67].

Practical architectures and statistical learning theory. As mentioned in Remarks 6.3 and 6.4, our DNN/DNO architectures difer from those typically employed in practice. A challenge of future work is to extend our analysis to practical architectures. In particular, for operator learning, we consider a type of PCA-Net, while FNOs and DeepONets are substantially more popular. The typical machinery to analyze generalization error with standard (as opposed to carefully handcrafted) architectures is statistical learning theory, which has been employed in [57, 67] and elsewhere to derive in-distribution generalization bounds for operator learning. Statistical learning theory considers the case of statistical noise whose variance may not be small, i.e., $e _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ in (4.1), and can be used to construct DNN/DNO estimators that achieve minimax optimal rates for nonparametric regression. Minimax rates concern the best achievable accuracy form m noisy samples uniformly over given class of functions or operators. Our work focuses on the more typical setting in scientific computing, of bounded, potentially adversarial noise that is small in norm. Thus, our results are more closely associated with optimal recovery rates, i.e., the best achievable accuracy over a given class from m noiseless samples. See [30] for discussion on minimax versus optimal recovery rates. A major distinction is that minimax rates typically decay no faster than the Monte Carlo rate, i.e., $\mathcal { O } ( m ^ { - 1 / 2 } )$ , while optimal recovery rates can be arbitrarily fast depending on the function or operator class. This is also reflected in our results, where the algebraic order can be arbitrarily large with suficient smoothness. Nonetheless, it is an interesting problem for future work to use statistical learning theory to derive OOD generalization guarantees for holomorphic functions and operators in the presence of statistical noise.

Structure beyond regularity. As mentioned in §1.5, holomorphy is a strong assumption which may not hold in practice. This has resulted in increasing discussion about what underlying structure may allow high-dimensional functions and operators to be eficiently learned [16,18,44,74]. Related to this is the question of how to enforce such structure into SciML models. For instance, regularization functionals have been used to encode PDE priors such as boundary conditions [56] and variational formulations [35]. Other approaches include PDE-informed architectures [15,34]. See also [69] and references therein. Much of this discussion centres on in-distribution generalization, but the same considerations apply OOD as well. Understanding what types of structures aid OOD generalization and, critically, how to enforce such structure in SciML models is a significant open problem.

Enhancing OOD performance. Similarly, there are many methods, largely empirical, that strive to improve OOD generalization in SciML. These include: [85], which studies incorporating physics, fine-tuning with new observations and multifidelity data; [19], which considers in-context learning; [36], which considers transfer learning; and [17, 70], which consider domain adaptation. See also [69] and references therein. It would be interesting to understand if any of these approaches conveyed a theoretical advantage over the generalization guarantees derived in this work.

Further types of OOD generalization. Finally, as mentioned in §1.5, there are diferent types of OOD generalization in SciML, especially in the realm of multioperator learning and foundation models. Our work considers OOD generalization in the sense of difering training and test distributions. Extending our work to other types of OOD generalization is an interesting topic for future research.

Acknowledgments. The authors would like to thank Chris Budd, Stefania Fresca, Anastasis Kratsios and Jakob Zech for helpful discussions.

## REFERENCES

[1] B. Adcock, S. Brugiapaglia, N. Dexter, and S. Moraga, Deep neural networks are efective at learning high-dimensional Hilbert-valued functions from limited data, in Proceedings of The Second Annual Conference on Mathematical and Scientific Machine Learning, J. Bruna, J. S. Hesthaven, and L. Zdeborov´a, eds., vol. 145 of Proc. Mach. Learn. Res. (PMLR), PMLR, 2021, pp. 1–36.

[2] B. Adcock, S. Brugiapaglia, N. Dexter, and S. Moraga, Learning smooth functions in high dimensions: from sparse polynomials to deep neural networks, in Numerical Analysis Meets Machine Learning, S. Mishra and A. Townsend, eds., vol. 25 of Handbook of Numerical Analysis, Elsevier, 2024, pp. 1–52.

[3] B. Adcock, S. Brugiapaglia, and C. G. Webster, Sparse Polynomial Approximation of High-Dimensional Functions, Comput. Sci. Eng., Society for Industrial and Applied Mathematics, Philadelphia, PA, 2022.

[4] B. Adcock, N. Dexter, and S. Moraga, Optimal approximation of infinite-dimensional holomorphic functions, Calcolo, 61 (2024), p. 12.

[5] B. Adcock, N. Dexter, and S. Moraga, Optimal deep learning of holomorphic operators between Banach spaces, in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, eds., vol. 37, Curran Associates, Inc., 2024, pp. 27725–27789.

[6] B. Adcock, M. Griebel, and G. Maier, The sample complexity oflearning Lipschitz operators with respect to Gaussian measures, arXiv:2410.23440, (2025).

[7] M. Arjovsky, Out of distribution generalization in machine learning, arXiv:2103.02667, (2021).

[8] J. Back, F. Nobile, L. Tamellini, and R. Tempone <sup>¨</sup> , Stochastic spectral Galerkin and collocation methods for PDEs with random coeficients: a numerical comparison, in Spectral and High Order Methods for Partial Diferential Equations, J. S. Hesthaven and E. M. Rønquist, eds., vol. 76 of Lect. Notes Comput. Sci. Eng., Berlin, Heidelberg, Germany, 2011, Springer, pp. 43–62.

[9] P. Batlle, M. Darcy, B. Hosseini, and H. Owhadi, Kernel methods are competitive for operator learning, J. Comput. Phys., 496 (2024), p. 112549.

[10] S. Ben-David, J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. W. Vaughan, A theory of learning from diferent domains, BIT, 79 (2010), pp. 151–175.

[11] S. Ben-David, J. Blitzer, K. Crammer, and F. Pereira, Analysis of representations for domain adaptation, in Advances in Neural Information Processing Systems, vol. 19, 2006, pp. 137–144.

[12] J. Blanchet and K. Murthy, Quantifying distributional model risk via optimal transport,

Math. Oper. Res., 44 (2019), pp. 565–600.

[13] A. Bonfanti, R. Santana, M. Ellero, and B. Gholami, On the generalization of PINNs outside the training domain and the hyperparameters influencing it, Neural Comput. Appl., 36 (2024), pp. 22677–22696.

[14] F. Bonnet, J. Mazari, P. Cinnella, and P. Gallinari, AirfRANS: High fidelity computational fluid dynamics dataset for approximating Reynolds-averaged Navier–Stokes solutions, in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 23463– 23478.

[15] N. Boulle, C. J. Earls, and A. Townsend<sup>´</sup> , Data-driven discovery of green’s functions with human-understandable deep learning, Scientific reports, 12 (2022), p. 4824.

[16] N. Boulle and A. Townsend<sup>´</sup> , A mathematical guide to operator learning, in Handbook of Numerical Analysis, vol. 25, Elsevier, 2024, pp. 83–125.

[17] S. Brivio, S. Fresca, and A. Manzoni, PTPI-DL-ROMs: Pre-trained physics-informed deep learning-based reduced order models for nonlinear parametrized PDEs, Comput. Methods Appl. Mech. Engrg., 432 (2024), p. 117404.

[18] S. Brugiapaglia, N. R. Franco, and N. H. Nelsen, A short tour of operator learning theory: convergence rates, statistical limits, and open questions, arXiv:2603.00819, (2026).

[19] W. Chen, J. Song, P. Ren, S. Subramanian, D. Morozov, and M. W. Mahoney, Dataeficient operator learning via unsupervised pretraining and in-context learning, in Proceedings of the 38th International Conference on Neural Information Processing Systems, Curran Associates Inc., 2024.

[20] A. Chkifa, A. Cohen, G. Migliorati, F. Nobile, and R. Tempone, Discrete least squares polynomial approximation with random evaluations - application to parametric and stochastic elliptic PDEs., ESAIM Math. Model. Numer. Anal., 49 (2015), pp. 815–837.

[21] A. Chkifa, A. Cohen, and C. Schwab, Breaking the curse of dimensionality in sparse polynomial approximation of parametric PDEs, J. Math. Pures Appl., 103 (2015), pp. 400–428.

[22] Y. Choi, S. W. Cheung, Y. Kim, P.-H. Tsai, A. N. Diaz, I. Zanardi, S. W. Chung, D. M. Copeland, C. Kendrick, W. Anderson, T. Iliescu, and M. Heinkenschloss, Defining Foundation Models for Computational Science: A Call for Clarity and Rigor, arXiv:2505.22904, (2025).

[23] M. Chu, Y. Liu, A. Biswas, and H.-W. Shen, Do physics foundation models learn generalizable physics? a bias-aware benchmark across physical regimes and distribution shifts, arXiv:2605.29283, (2026).

[24] A. Cohen and R. A. DeVore, Approximation of high-dimensional parametric PDEs, Acta Numer., 24 (2015), pp. 1–159.

[25] A. Cohen, R. A. DeVore, and C. Schwab, Analytic regularity and polynomial approximation of parametric and stochastic elliptic PDE’s, Anal. Appl. (Singap.), 9 (2011), pp. 11–47.

[26] A. Cohen, G. Migliorati, and F. Nobile, Discrete least-squares approximations over optimized downward closed polynomial spaces in arbitrary dimension, Constr. Approx., 45 (2017), pp. 497–519.

[27] M. V. de Hoop, N. B. Kovachki, N. H. Nelsen, and A. M. Stuart, Convergence Rates for Learning Linear Operators from Noisy Data, SIAM/ASA J. Uncertain. Quantif., 11 (2023), pp. 480–513.

[28] L. Demanet and A. Townsend, Stable extrapolation of analytic functions, Found. Comput. Math., 19 (2019), pp. 297–331.

[29] R. DeVore, B. Hanin, and G. Petrova, Neural network approximation, Acta Numer., 30 (2021), pp. 327–444.

[30] R. DeVore, R. D. Nowak, R. Parhi, G. Petrova, and J. W. Siegel, Optimal recovery meets minimax estimation, arXiv:2502.17671, (2025).

[31] R. A. DeVore, Nonlinear approximation, Acta Numer., 7 (1998), pp. 51–150.

[32] D. Elbrachter, D. Perekrestenko, P. Grohs, and H. B <sup>¨</sup> olcskei <sup>¨</sup> , Deep neural network approximation theory, IEEE Trans. Inform. Theory, 67 (2021), pp. 2581–2623.

[33] L. Fesser, L. D’Amico-Wong, and R. Qiu, Understanding and Mitigating Extrapolation Failures in Physics-Informed Neural Networks, arXiv:2306.09478, (2023).

[34] C. R. Gin, D. E. Shea, S. L. Brunton, and J. N. Kutz, Deepgreen: deep learning of green’s functions for nonlinear boundary value problems, Scientific reports, 11 (2021), p. 21614.

[35] S. Goswami, A. Bora, Y. Yu, and G. E. Karniadakis, Physics-informed deep neural operator networks, in Machine learning in modeling and simulation: methods and applications, Springer, 2023, pp. 219–254.

[36] S. Goswami, K. Kontolati, M. D. Shields, and G. E. Karniadakis, Deep transfer operator learning for partial diferential equations under conditional shift, Nat. Mach. Intell., 4 (2022), pp. 1155–1164.

[37] N. Guerra, N. H. Nelsen, and Y. Yang, Learning where to learn: training data distribution optimization for scientific machine learning, arXiv:2505.21626, (2025).

[38] I. Guhring and M. Raslan <sup>¨</sup> , Approximation rates for neural networks with encodable weights in smoothness spaces, Neural Networks, 134 (2021), pp. 107–130.

[39] M. Herde, B. Raonic, T. Rohner, R. K <sup>´</sup> appeli, R. Molinaro, E. de B <sup>¨</sup> ezenac, and<sup>´</sup> S. Mishra, Poseidon: Eficient foundation models for PDEs, in Advances in Neural Information Processing Systems, vol. 37, 2024, pp. 72525–72624.

[40] L. Herrmann, C. Schwab, and J. Zech, Neural and spectral operator surrogates: Unified construction and expression rate bounds, Adv. Comput. Math., 50 (2024).

[41] R. Hong and A. Kratsios, Bridging the gap between approximation and learning via optimal approximation by ReLU MLPs of maximal regularity, arXiv:2409.12335, (2024).

[42] S. Hou, P. Kassraie, A. Kratsios, A. Krause, and J. Rothfuss, Instance-dependent generalization bounds via optimal transport, J. Mach. Learn. Res., 24 (2023), pp. 1–51.

[43] S. Jiang and A. F. Queiruga, What should physics foundation model benchmarks measure?, Comput. Sci. Eng., (2026), pp. 1–7.

[44] N. B. Kovachki, S. Lanthaler, and A. M. Stuart, Operator learning: algorithms and analysis, in Numerical Analysis Meets Machine Learning, S. Mishra and A. Townsend, eds., vol. 25 of Handbook of Numerical Analysis, Elsevier, 2024, pp. 419–467.

[45] A. Kratsios, S. Brugiapaglia, B. J. Kim, G. Cousins, and H. S. d. O. Borde, Algorithmic foundations of deep learning: Complexity-theoretic rates and a characterization of universal approximation, arXiv preprint arXiv:2606.26705, (2026).

[46] D. Krueger, E. Caballero, J.-H. Jacobsen, A. Zhang, J. Binas, D. Zhang, R. Le Priol, and A. Courville, Out-of-distribution generalization via risk extrapolation (REx), in Proceedings of the 38th International Conference on Machine Learning, M. Meila and T. Zhang, eds., vol. 139 of Proceedings of Machine Learning Research, PMLR, 2021, pp. 5815–5826.

[47] D. Kuhn, S. Shafiee, and W. Wiesemann, Distributionally robust optimization, Acta Numer., 34 (2025), pp. 579–804.

[48] S. Lanthaler, Operator learning with PCA-Net: upper and lower complexity bounds, J. Mach. Learn. Res., 24 (2023), pp. 318:15014–318:15080.

[49] S. Lanthaler, S. Mishra, and G. E. Karniadakis, Error estimates for DeepOnets: A deep learning framework in infinite dimensions, arXiv:2102.09618, (2022).

[50] S. Lanthaler and A. M. Stuart, The parametric complexity of operator learning, arXiv:2306.15924, (2024).

[51] J. A. Lara Benitez, T. Furuya, F. Faucher, A. Kratsios, X. Tricoche, and M. V. De Hoop, Out-of-distributional risk bounds for neural operators with applications to the Helmholtz equation, J. Comput. Phys., 513 (2024), p. 113168.

[52] J. Lee and M. Raginsky, Minimax statistical learning with Wasserstein distances, in Advances in Neural Information Processing Systems, vol. 31, 2018.

[53] A. Levine and S. Feizi, Wasserstein smoothing: Certified robustness against Wasserstein adversarial attacks, in Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, S. Chiappa and R. Calandra, eds., vol. 108 of Proceedings of Machine Learning Research, PMLR, 2020, pp. 3938–3947.

[54] K. Li, A. N. Rubungo, X. Lei, D. Persaud, K. Choudhary, B. DeCost, A. B. Dieng, and J. Hattrick-Simpers, Probing out-of-distribution generalization in machine learning for materials, Communications Materials, 6 (2025), p. 9.

[55] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar, Fourier neural operator for parametric partial diferential equations, in ICLR, 2021.

[56] Z. Li, H. Zheng, N. B. Kovachki, D. Jin, H. Chen, B. Liu, K. Azizzadenesheli, and A. Anandkumar, Physics-informed neural operator for learning partial diferential equations, ACM/IMS J. Data Sci., 1 (2024), pp. 1–27.

[57] H. Liu, H. Yang, M. Chen, T. Zhao, and W. Liao, Deep nonparametric estimation of operators between infinite dimensional spaces, J. Mach. Learn. Res., 25 (2024), pp. 1–67.

[58] J. Liu, Z. Shen, Y. He, X. Zhang, R. Xu, H. Yu, and P. Cui, Towards out-of-distribution generalization: A survey, arXiv:2108.13624, (2023).

[59] J. Lu, Z. Shen, H. Yang, and S. Zhang, Deep network approximation for smooth functions, SIAM J. Math. Anal., 53 (2021), pp. 5465–5506.

[60] L. Lu, P. Jin, Z. Pang, G. Zhang, and G. E. Karniadakis, Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators, Nat. Mach. Intell., 3 (2021), pp. 218–229.

[61] G. Migliorati, Adaptive approximation by optimal weighted least squares methods, SIAM J. Numer. Anal, 57 (2019), pp. 2217–2245.

[62] E. S. Muckley, J. E. Saal, B. Meredig, C. S. Roper, and J. H. Martin, Interpretable models for extrapolation in scientific machine learning, Digit. Discov., 2 (2023), pp. 1425– 1435.

[63] B. D. Nguyen and S. Sandfeld, Out-of-distribution generalization of deep-learning surrogates for 2D PDE-generated dynamics in the small-data regime, arXiv:2601.08404, (2026).

[64] F. Nobile, R. Tempone, and C. G. Webster, An anisotropic sparse grid stochastic collocation method for partial diferential equations with random input data, SIAM J. Numer. Anal., 46 (2008), pp. 2411–2442.

[65] F. Nobile, R. Tempone, and C. G. Webster, A sparse grid stochastic collocation method for partial diferential equations with random input data, SIAM J. Numer. Anal., 46 (2008), pp. 2309–2345.

[66] J. A. A. Opschoor, C. Schwab, and J. Zech, Exponential ReLU DNN expression of holomorphic maps in high dimension, Constr. Approx., 55 (2022), pp. 537–582.

[67] N. Reinhardt, S. Wang, and J. Zech, Statistical learning theory for neural operators, arXiv:2412.17582, (2024).

[68] C. Schwab and J. Zech, Deep learning in high dimension: neural network expression rates for generalized polynomial chaos expansions in UQ, Anal. Appl. (Singap.), 17 (2019), pp. 19– 55.

[69] L. Serrano, J. Han, E. Oyallon, S. Ho, and R. Morel, Test-time Generalization for Physics through Neural Operator Splitting, arXiv:2602.00884, (2026).

[70] P. Setinek, G. Galletti, T. Gross, D. Schnurer, J. Brandstetter, and W. Zellinger<sup>¨</sup> , SIMSHIFT: A benchmark for adapting neural surrogates to distribution shifts, arXiv:2506.12007, (2025).

[71] L. Shikhman, Diagnosing failure modes of neural operators across diverse PDE families, arXiv:2601.11428, (2026).

[72] A. Sinha, H. Namkoong, and J. C. Duchi, Certifying some distributional robustness with principled adversarial training, in International Conference on Learning Representations, 2018.

[73] U. Subedi and A. Tewari, Is Zero-Shot Super-Resolution Possible in Operator Learning?, arXiv:2606.00296, (2026).

[74] U. Subedi and A. Tewari, Operator learning: a statistical perspective, Annu. Rev. Stat. Appl., 13 (2026), pp. 123–148.

[75] S. Surjanovic and D. Bingham, Virtual library of simulation experiments: test functions and datasets. http://www.sfu.ca/<sup>∼</sup>ssurjano.

[76] L. N. Trefethen, Quantifying the ill-conditioning of analytic continuation, BIT, 60 (2020), pp. 901–915.

[77] H. Wang et al., Scientific discovery in the age of artificial intelligence, Nature, 620 (2023), pp. 47–60.

[78] S. Wang and P. Perdikaris, Long-time integration of parametric evolution equations with physics-informed DeepONets, J. Comput. Phys., 475 (2023), p. 111855.

[79] J. Westermann, B. Huber, T. O’Leary-Roseberry, and J. Zech, Performance of neural and polynomial operator surrogates, arXiv:2604.00689, (2026).

[80] K. Xu, M. Zhang, J. Li, S. S. Du, K.-I. Kawarabayashi, and S. Jegelka, How neural networks extrapolate: From feedforward to graph neural networks, in International Conference on Learning Representations, 2021.

[81] Y. Yang, Z. Zhang, W. Zhu, W. Liao, and H. Liu, Generalization guarantees for multi-input neural operator learning in Sobolev spaces, arXiv:2606.17419, (2026).

[82] D. Yarotsky, Error bounds for approximations with deep ReLU networks, Neural Networks, 94 (2017), pp. 103–114.

[83] L. Yuan, H. S. Park, and E. Lejeune, Towards out of distribution generalization for problems in mechanics, Comput. Methods Appl. Mech. Engrg., 400 (2022), p. 115569.

[84] X. Zhang et al, Artificial intelligence for science in quantum, atomistic, and continuum systems, Foundations and Trends in Machine Learning, 18 (2025), pp. 385–849.

[85] M. Zhu, H. Zhang, A. Jiao, G. E. Karniadakis, and L. Lu, Reliable extrapolation of deep neural operators informed by physics or sparse observations, Comput. Methods Appl. Mech. Engrg., 412 (2023), p. 116064.

# SUPPLEMENTARY MATERIALS: INTO THE DANGER ZONE:STABLE EXTRAPOLATION IN HIGH-DIMENSIONAL FUNCTIONAND OPERATOR LEARNING

## BEN ADCOCK<sup>†</sup>, SIMONE BRUGIAPAGLIA<sup>‡</sup>, AND XUEMENG WANG<sup>†</sup>

SM1. Proofs of the main results. In this appendix, we present the proofs of the main results from the paper.

Overall, these proofs use, adapt and extend tools from the approximation theory of infinite-dimensional holomorphic functions [SM16,SM17,SM13,SM6,SM15], in combination with emulation techniques [SM5,SM18,SM20,SM21,SM24,SM33,SM31, SM25,SM28,SM32] to establish results for learning DNNs and DNOs. We commence in §SM1.1 with some additional notation and concepts. §SM1.2 presents what is arguably the key technical innovation in this work, namely, new weighted summability bounds (Lemma SM1.4 and SM1.5) and weighted k-term approximation error bounds (Theorems SM1.2 and SM1.3) for holomorphic functions for exponentially growing weights. As we explain, summability with respect to exponentially growing weights is directly related to extrapolation with polynomials.

In §SM1.3 we use these results to prove our main result on the existence of good polynomial estimators for extrapolation, Theorem 3.2. Next, in §SM1.4 we prove the main result on sample-eficient out-of-distribution learning via polynomials, Theorem 4.1. Finally, in §SM1.5 we establish the main results for DNNs and DNNOs, Theorems 4.4 and 5.6, respectively.

SM1.1. Additional notation and concepts. While the results in the main paper consider real-valued functions $f : D \to \mathbb { R }$ , in our proofs we will establish results for Hilbert-valued functions $f : D  \mathcal { V }$ , where $( \mathcal { V } , \langle \cdot , \cdot \rangle _ { \mathcal { V } } )$ is a Hilbert space. This not only generalizes the results in the main paper, but is also needed in order to establish the results on operator learning considered in §5.

This necessitates some additional concepts (see, e.g., [SM6, SM8] and references therein). First, given a set D with a measure ϱ and $1 \leq p \leq \infty$ , we define the Lebesgue–Bochner space $L _ { \varrho } ^ { p } ( D ; \mathcal { V } )$ as the space consisting of (equivalence classes of) strongly ϱ-measurable functions $f : D  \mathcal { V }$ for which $\| f \| _ { L _ { \varrho } ^ { p } ( D ; \mathcal { V } ) } < \infty$ , where

$$
\begin{array}{c} \begin{array} { r } { \| f \| _ { L _ { \varrho } ^ { p } ( D ; \mathcal { Y } ) } : = \left\{ \left( \int _ { D } \| f ( \pmb { y } ) \| _ { \mathcal { Y } } ^ { p } \mathrm { d } \varrho ( \pmb { y } ) \right) ^ { 1 / p } \quad 1 \le p < \infty , \right.} \\ { \mathrm { e s s } \operatorname* { s u p } _ { \pmb { y } \in D } \| f ( \pmb { y } ) \| _ { \mathcal { Y } } \qquad p = \infty . } \end{array}   \end{array}
$$

For simplicity we write $L _ { \varrho } ^ { p } ( D )$ when $\mathcal { V } = \mathbb { R }$

Second, we require various notions relating to sequences whose entries take values in Y. Let $d \in \mathbb { N } \cup \{ \infty \} , \Lambda \subseteq \mathcal { F }$ be a multi-index set and $\pmb { c } = ( c _ { n } ) _ { n \in \Lambda }$ be a sequence with Y-valued entries. For $0 < p \le \infty$ , we write $\ell ^ { p } ( \Lambda ; \mathcal { V } )$ for the space of Y-valued

sequences $\pmb { c } = ( c _ { n } ) _ { n \in \Lambda }$ for which $\| \boldsymbol { c } \| _ { p ; \mathcal { V } } < \infty$ , where

$$
\left\| { \pmb { c } } \right\| _ { p ; \mathscr { V } } = \left\{ \left( \sum _ { \pmb { n } \in \Lambda } \left\| { \pmb { c _ { n } } } \right\| _ { \mathscr { V } } ^ { p } \right) ^ { \frac { 1 } { p } } \quad 0 < p < \infty , \right.
$$

When $\mathcal { V } = \mathbb { R }$ , we just write $\ell ^ { p } ( \Lambda )$ and $\| \cdot \| _ { p } .$ . Note that $\| \cdot \| _ { p ; \mathcal { y } }$ defines a norm when $1 \leq p \leq \infty$ and quasi-norm when $0 < p < \mathrm { \hat { 1 } }$ . Now let $\pmb { c } \in \mathring { \ell ^ { p } } ( \Lambda ; \mathcal { V } )$ and $s \in \mathbb N$ with $1 \leq s \leq | \Lambda |$ . We define the ℓ<sup>p</sup>-norm best s-term approximation error of c as

$$
\begin{array} { r } { \sigma _ { s } ( c ) _ { p ; \mathcal { V } } = \operatorname* { i n f } \left\{ \left\| c - z \right\| _ { p ; \mathcal { V } } \colon z \in \ell ^ { p } ( \Lambda ; \mathcal { V } ) , ~ | \mathrm { s u p p } ( z ) | \le s \right\} . } \end{array}\tag{SM1.1}
$$

Here,

$$
\operatorname { s u p p } ( z ) = \{ { \pmb { n } } \in \Lambda : z _ { { \pmb { n } } } \neq 0 \}
$$

is the support of $_ { z , }$ . Given a Y-valued sequence c and $S \subseteq \Lambda$ , we write $c _ { S }$ for the Y-valued vector with nth entry equal to $c _ { n } { \mathrm { ~ i f ~ } } n \in S$ and zero otherwise. Observe that

$$
\begin{array} { r } { \sigma _ { s } ( c ) _ { p ; \mathcal { V } } = \operatorname* { i n f } \left\{ \left\| c - c _ { S } \right\| _ { p ; \mathcal { V } } : S \subseteq \Lambda , \ | S | \leq s \right\} . } \end{array}
$$

Now let ${ \pmb w } = ( w _ { \pmb { n } } ) _ { { \pmb n } \in \Lambda } > { \pmb 0 }$ be a vector of positive weights. For $1 \le p \le 2$ , we define the weighted $\ell _ { w } ^ { p } ( \Lambda ; \mathcal { V } )$ space as the space of Y-valued sequences $\pmb { c } = ( c _ { n } ) _ { n \in \Lambda }$ for which $\| \pmb { c } \| _ { p , \pmb { w } ; \mathcal { V } } < \infty$ , where

$$
\| c \| _ { p , w ; \mathcal { V } } = \left( \sum _ { n \in \Lambda } w _ { n } ^ { 2 - p } \| c _ { n } \| _ { \mathcal { V } } ^ { p } \right) ^ { \frac { 1 } { p } } , \quad 1 \le p \le 2 .
$$

Notice that $\begin{array} { r l } { \| \cdot \| _ { p , w ; \mathcal { V } } } \end{array}$ coincides with the unweighted norm $\| \cdot \| _ { p ; \mathcal { Y } }$ for any w when $p = 2$ or for any p when ${ \pmb w } = { \bf 1 }$ . Next, we define the weighted cardinality of an index set $S \subseteq \Lambda$ as $\begin{array} { r } { | S | _ { w } = \sum _ { n \in S } w _ { n } ^ { 2 } } \end{array}$ and, for $0 < k \leq | \Lambda | _ { w }$ , we define the $\ell _ { w } ^ { p }$ -norm weighted best $( k , w )$ -term approximation error of a Y-valued vector or sequence $\pmb { c } \in \ell _ { \pmb { w } } ^ { p } ( \Lambda ; \mathcal { V } )$ as

$$
\begin{array} { r } { \sigma _ { k } ( \pmb { c } ) _ { p , \pmb { w } ; \mathscr { V } } = \operatorname* { i n f } \left\{ \| \pmb { c } - \pmb { z } \| _ { p , \pmb { w } ; \mathscr { V } } : \pmb { z } \in \ell _ { \pmb { w } } ^ { p } ( \Lambda ; \mathscr { V } ) , \ | \mathrm { s u p p } ( \pmb { z } ) | _ { \pmb { w } } \leq k \right\} . } \end{array}\tag{SM1.2}
$$

For later use, we also require the following result, which is a weighted version of Stechkin’s inequality. See [SM6, Lem. 3.12] (note the result therein is only stated for $\mathcal { V } = \mathbb { R }$ , but it readily extends to Hilbert-valued sequences).

Lemma SM1.1 (Weighted Stechkin’s inequality). $L e t d \in \mathbb { N } \cup \{ \infty \} , 0 < p \leq 2 \leq$ 2, $k > 0 , c \in \ell _ { w } ^ { p } ( \Lambda ; \mathcal { V } )$ for some $\Lambda \subseteq \mathbb { N } _ { 0 } ^ { d }$ and ${ \pmb w } = ( w _ { \pmb { n } } ) _ { \pmb { n } \in \Lambda } > { \pmb 0 }$ . Then

$$
\begin{array} { r } { \sigma _ { k } ( \pmb { c } ) _ { q , w ; \mathcal { V } } \leq \| \pmb { c } \| _ { p , w ; \mathcal { V } } k ^ { \frac { 1 } { q } - \frac { 1 } { p } } . } \end{array}
$$

Finally, we also need several concepts involving lower and anchored sets. For $i \in [ d ]$ let $e _ { i }$ be the multi-index with ith entry equal to 1 and all other entries equal to zero. A set $\Lambda \subseteq \mathbb { N } _ { 0 } ^ { d } .$ , where $d \in \mathbb { N } \cup \{ \infty \}$ , is lower if, whenever $\pmb { n } \in \Lambda$ and ${ \mathfrak { n } } ^ { \prime } \leq n$ , it also holds that $\mathbf { { \boldsymbol { n } } } ^ { \prime } \in \Lambda$ . Moreover, Λ is anchored if it is lower and if, whenever $e _ { j } \in \Lambda$ for some $j \in [ d ]$ , it also holds that $\{ e _ { 1 } , e _ { 2 } , \dotsc , e _ { j } \} \subseteq \Lambda$ . A scalar-valued sequence $\pmb { d } = ( d _ { n } ) _ { n \in \Lambda }$ is monotonically nonincreasing if $d _ { n } \geq d _ { n ^ { \prime } }$ whenever $\mathbf { \Delta } n \leq { \mathbf { \mu } } n ^ { \prime }$ . It is anchored if it is monotonically nonincreasing and $d _ { e _ { j } } \ \leq \ d _ { e _ { i } }$ whenever $i , j \in [ d ]$ with $i \leq j$ . Now let $c \in \ell ^ { \infty } ( \Lambda ; \mathcal { V } )$ . An anchored majorant of c is any scalar-valued sequence $\pmb { d } = ( d _ { n } ) _ { n \in \Lambda }$ that is anchored and satisfies $d _ { n } \geq \| c _ { n } \| _ { \mathcal { V } } , \forall n \in \Lambda$ . The minimal anchored majorant of c is the (unique) scalar-valued sequence c˜ that is an anchored majorant and for which $\tilde { c } \leq d$ for any other anchored majorant d. It has the explicit expression

$$
\tilde { c } _ { n } = \left\{ \begin{array} { l l } { \operatorname* { s u p } \{ \| c _ { \mu } \| _ { \mathcal { V } } : \mu \geq n \} } & { \mathrm { ~ i f ~ } n \not = e _ { j } \mathrm { ~ f o r ~ a n y ~ } j \in [ d ] , } \\ { \operatorname* { s u p } \{ \| c _ { \mu } \| _ { \mathcal { V } } : \mu \geq e _ { i } \mathrm { ~ f o r ~ s o m e ~ } i \geq j \} } & { \mathrm { ~ i f ~ } n = e _ { j } \mathrm { ~ f o r ~ s o m e ~ } j \in [ d ] . } \end{array} \right.\tag{SM1.3}
$$

Given $0 < p \leq \infty$ , we define the anchored $\ell ^ { p \ }$ space $\ell _ { \mathsf { A } } ^ { p } ( \Lambda ; \mathcal { V } )$ as the space of sequences $c \in \ell ^ { \infty } ( \Lambda ; \mathcal { V } )$ for which $\tilde { { \boldsymbol { c } } } \in \ell ^ { p } ( \Lambda )$ , and define the (quasi-)norm $\| \pmb { c } \| _ { p , \mathsf { A } ; \mathcal { V } } = \| \widetilde { \pmb { c } } \| _ { p } .$ Finally, for $1 \leq s \leq | \Lambda |$ we also define the ℓ<sup>p</sup>-norm best s-term approximation error in anchored sets by

$$
\begin{array} { r } { \sigma _ { s , \mathsf { A } } ( c ) _ { p ; \mathcal { V } } = \operatorname* { i n f } \left\{ \| c - z \| _ { p ; \mathcal { V } } : z \in \ell ^ { p } ( \Lambda ; \mathcal { V } ) , \ | \operatorname { s u p p } ( z ) | \leq s , \operatorname { s u p p } ( z ) \ \mathrm { a n c h o r e d } \right\} . } \end{array}\tag{SM1.4}
$$

SM1.2. Summability and weighted k-term approximation of polynomial coeficients. In this section, we provide several results on weighted summability and weighted k-term approximation of polynomial coeficients of $( b , \varepsilon ; \mathcal { V } )$ -holomorphic functions. They are crucial steps towards establishing our main results, as they can be used to estimate both the in-distribution and out-of-distribution polynomial approximation errors.

Unweighted summability of polynomial coeficients of $( b , \varepsilon ; \mathcal { V } )$ -holomorphic functions is a topic at the core of infinite-dimensional polynomial approximation theory [SM6, SM15] since, in view of Stechkin’s inequality, it implies algebraic convergence of the best s-term approximation. The extension from unweighted to weighted summability and weighted k-term approximation was a crucial advance, since sets with low weighted cardinality are critical to both guaranteeing convergence in the $L ^ { \infty } \mathrm { - n o r m }$ and to ensuring that sample-eficient estimators can be constructed via least-squares or compressed sensing techniques. See [SM30, SM14, SM29], as well as [SM6,SM4,SM8]. However, these works consider only the in-distribution error and weights that grow algebraically fast. As we explain next, to address out-of-distribution errors, we require summability of polynomial coeficients with respect to exponentially growing weights.

Recall that $\varrho$ is the uniform probability measure on $[ - 1 , 1 ] ^ { \mathbb { N } }$ . To commence, observe first that

$$
\| \Psi _ { \pmb { n } } \| _ { L _ { \varrho } ^ { \infty } ( \mathbb { R } ^ { n } ) } = \prod _ { j \in \mathrm { s u p p } ( n ) } \sqrt { 2 n _ { j } + 1 } = : u _ { \pmb { n } } , \quad \forall \pmb { n } \in \mathscr { F } ,\tag{SM1.5}
$$

which follows directly from the definition of the $\Psi _ { n }$ . Next, we consider the behaviour of the Legendre polynomials on $D _ { \omega }$ , where $D _ { \omega }$ is as in (1.3). For this, we use the integral representation

$$
P _ { n } ( x ) = \frac { 1 } { \pi } \int _ { 0 } ^ { \pi } ( x + \sqrt { x ^ { 2 } - 1 } \cos ( \theta ) ) ^ { n } \mathrm { d } \theta , \quad \forall | x | > 1 , \ n \in \mathbb { N } _ { 0 } .
$$

which follows from the contour integral representation of $P _ { n } ~ [ \mathrm { S M 1 }$ , Tab. 18.10.1]. Hence

$$
| P _ { n } ( x ) | \leq \zeta ( \omega ) ^ { n } , \quad \forall | x | \leq \omega , \ \omega > 1 , \ n \in \mathbb { N } _ { 0 } .
$$

We deduce that, for any probability measure $\mu$ supported in $D _ { \omega }$

$$
\| \Psi _ { n } \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { n } ) } \leq \prod _ { j \in \operatorname { s u p p } ( n ) } \sqrt { 2 n _ { j } + 1 } \zeta ( \omega _ { j } ) ^ { n _ { j } } = : w _ { n } , \quad \forall n \in \mathcal { F } .\tag{SM1.6}
$$

The key consequence of these observations is the following. Let $f : D  \mathcal { V }$ with Y-valued coeficients $\pmb { c } = ( c _ { n } ) _ { \pmb { n } \in \mathcal { F } } , S \subset \mathcal { F }$ and $\begin{array} { r } { f _ { S } = \sum _ { n \in S } c _ { n } \Psi _ { n } } \end{array}$ . Then observe that the in-distribution $L ^ { \infty }$ -norm error can be bounded by

$$
\| f - f _ { S } \| _ { L _ { \varrho } ^ { \infty } ( D ; \mathcal { V } ) } \leq \| c - c _ { S } \| _ { 1 , { \pmb { u } } ; \mathcal { V } }
$$

where $\pmb { u } = ( u _ { n } ) _ { n \in \mathcal { F } }$ , and the out-of-distribution $L ^ { \infty }$ -norm error can be bounded by

$$
\begin{array} { r } { \| \boldsymbol { f } - \boldsymbol { f } _ { S } \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } \leq \| \boldsymbol { c } - \boldsymbol { c } _ { S } \| _ { 1 , w ; \mathcal { V } } , } \end{array}
$$

where ${ \pmb w } = ( w _ { \pmb { n } } ) _ { \pmb { n } \in \mathcal { F } }$ . Hence, the weighted Stechkin inequality (Lemma SM1.1) implies algebraic decay of these errors, whenever $\pmb { c } \in \mathcal { l } _ { \pmb { u } } ^ { p } ( \mathcal { F } ; \mathcal { V } )$ or $\pmb { c } \in \ell _ { \pmb { w } } ^ { p } ( \mathcal { F } ; \mathcal { V } )$ , respectively. We now establish conditions under which these properties hold (note that the former is just a special case of the latter corresponding to $\omega = \mathbf { 1 } )$ The key innovation here is we show weighted summability of the polynomial coeficients of a $( b , \varepsilon ; \mathcal { V } )$ -holomorphic function with respect to the exponentially-growing weights w. Our first main result in this section is the following.

Theorem SM1.2 (Weighted summability and weighted k-term approximation with exponentially growing weights). Let $\omega \geq 1$ and suppose that $f \in \mathcal { H } ( b , \varepsilon ; \mathcal { V } )$ where $\pmb { b } \in \ell ^ { 1 } ( \mathbb { N } )$ satisfies

$$
b \odot r _ { p } \in \ell ^ { p } ( \mathbb { N } ) , \quad \left\| b \odot ( r _ { p } - \mathbf { 1 } ) \right\| _ { 1 } < \varepsilon , \qquad w h e r e \ r _ { p } = \zeta ( \omega ) ^ { 2 / p - 1 }\tag{SM1.7}
$$

for some $0 < p < 1$ . Then its coeficients $\pmb { c } \in \ell _ { \pmb { w } } ^ { p } ( \mathcal { F } ; \mathcal { V } )$ , where w is given by (SM1.6). Moreover, for any $k > 0$ , there exists a set $S \subset { \mathcal { F } }$ depending on $k , b , \varepsilon$ and ω only with $| S | _ { w } \le k$ such that

$$
\begin{array} { r } { \| \boldsymbol { c } - \boldsymbol { c } _ { S } \| _ { q , w ; \mathcal { V } } \le C ( \boldsymbol { b } , \boldsymbol { \varepsilon } , \omega , p ) k ^ { 1 / q - 1 / p } , } \end{array}
$$

for any $q \in ( p , 2 ]$

Secondly, we also establish a result on approximation of such an $f$ via anchored sets, which will be needed in order to establish the results in §4 and §5 of the paper. Here, we recall the definition of monotone ℓ<sup>p</sup>-space, $\ell _ { \mathsf { M } } ^ { p } ( \mathbb { N } )$ (see §1.2).

Theorem SM1.3 (Weighted summability and anchored s-term approximation with exponentially growing weights). Let $\omega \geq 1$ and suppose that $f \in \mathcal { H } ( b , \varepsilon ; \mathcal { V } )$ where $\pmb { b } \in \ell ^ { 1 } ( \mathbb { N } )$ satisfies

$$
b \odot r _ { p } \in \ell _ { \mathsf { M } } ^ { p } ( \mathbb { N } ) , \quad \left\| b \odot ( r _ { p } - \mathbf { 1 } ) \right\| _ { 1 } < \varepsilon , \qquad w h e r e \ r _ { p } = \zeta ( \omega ) ^ { 2 / p - 1 }\tag{SM1.8}
$$

for some $0 < p < 1$ . Then its coeficients $\pmb { c } \in \ell _ { \pmb { w } } ^ { p } ( \mathcal { F } ; \mathcal { V } )$ , where w is given by (SM1.6). Moreover, for any $s \in \mathbb N$ , there exists an anchored set $S \subset { \mathcal { F } }$ depending on s, b, ε and ω only with $| S | \le s$ such that

$$
\begin{array} { r } { \| c - c _ { S } \| _ { q , w ; \mathcal { V } } \le C ( b , \varepsilon , \omega , p ) s ^ { 1 / q - 1 / p } , } \end{array}
$$

for any $q \in ( p , 2 ]$

To prove these results, we require the following two abstract summability lemmas. Results of this type have a long history in the literature, first appearing in [SM13, SM15, SM16] in the context of Chebyshev and Legendre polynomial coeficients of, firstly, parameter-to-solution maps of parametric PDEs and, latterly, $( b , \varepsilon ; \mathcal { V } )$ -holomorphic functions [SM6, Thms. $3 . 2 8 \mathrm { ~ \ } \& \mathrm { ~ \ } 3 . 3 3 ]$ These two lemmas are based directly on [SM8, Lem. 5.5], which considered abstract sequences of coeficients, but only algebraically growing weights.

Lemma SM1.4 (Abstract summability). Let $0 < p < 1 , \varepsilon > 0 , \xi : ( 1 , \infty ) \to$ $[ 0 , \infty )$ be continuous with $\xi ( t )$ bounded as $t \to \infty , r \geq { \bf 1 }$ and $\pmb { b } \in \ell ^ { 1 } ( \mathbb { N } )$ be such that

$$
b \odot { \pmb r } \in \ell ^ { p } ( \mathbb { N } ) , \qquad \| { \pmb b } \odot ( { \pmb r } - { \bf 1 } ) \| _ { 1 } < \varepsilon .\tag{SM1.9}
$$

Let $c , \gamma > 0$ be constant and $\mathbf { \psi } _ { d } ~ = ~ ( d _ { n } ) _ { n \in \mathcal { F } }$ be such that $| d _ { \bf 0 } | \le c$ and, for every ${ \pmb { n } } \in \mathcal { F } \backslash \{ { \bf 0 } \}$

$$
| d _ { \pmb { n } } | \leq \prod _ { k \in \mathrm { s u p p } ( \pmb { n } ) } \xi ( \rho _ { k } ) \rho _ { k } ^ { - n _ { k } } r _ { k } ^ { n _ { k } } ( c n _ { k } + 1 ) ^ { \gamma }
$$

for all $\pmb { \rho } = ( \rho _ { j } ) _ { j \in \mathbb { N } } \geq \mathbf { 1 }$ satisfiying $\{ k : \rho _ { k } > 1 \} \supseteq \mathrm { s u p p } ( n )$ and

$$
\sum _ { j = 1 } ^ { \infty } \left( \frac { \rho _ { j } + \rho _ { j } ^ { - 1 } } { 2 } - 1 \right) b _ { j } \leq \varepsilon .\tag{SM1.10}
$$

Then $d \in \ell ^ { p } ( { \mathcal { F } } )$ and $\left\| d \right\| _ { p } \leq C ( b , \varepsilon , r , p , c , \gamma , \xi )$

Proof. We follow the setup of [SM8, Lem. 5.5]. We split $\mathbb { N } = E \cup F$ , where $E = [ d ]$ and $F = \mathbb { N } \backslash [ d ]$ , where $d \in \mathbb { N }$ will be chosen later. By assumption, there exists an $0 < s < 1$ such that $\| \pmb { b } \odot ( \pmb { r } - \pmb { 1 } ) \| _ { 1 } \le s ^ { 2 } \varepsilon$ . Now let ${ \pmb a } = ( a _ { j } ) _ { j \in \mathbb { N } } > { \bf 1 }$ be a sequence that will be specified later that satisfies

$$
\sum _ { j = 1 } ^ { \infty } ( a _ { j } - 1 ) b _ { j } \leq s \varepsilon\tag{SM1.11}
$$

Given $\pmb { n } \in \mathcal { F }$ , let $\tilde { \rho } = \tilde { \rho } ( b , \varepsilon , n ) \ge 1$ be defined by

$$
\frac { \tilde { \rho } _ { j } + \tilde { \rho } _ { j } ^ { - 1 } } { 2 } = \left\{ \begin{array} { l l } { a _ { j } } & { j \in \mathrm { s u p p } ( { \pmb n } _ { E } ) } \\ { a _ { j } + \frac { ( 1 - s ) \varepsilon n _ { j } } { b _ { j } \| { \pmb n _ { F } } \| _ { 1 } } } & { j \in \mathrm { s u p p } ( { \pmb n _ { F } } ) } \\ { 1 } & { j \notin \mathrm { s u p p } ( { \pmb n } ) } \end{array} \right.
$$

Observe that $\{ k : \rho _ { k } > 1 \} = \mathrm { s u p p } ( n )$ and $\tilde { \rho }$ satisfies (SM1.10). Now suppose that

$$
a _ { \operatorname* { m i n } } = \operatorname* { i n f } _ { j } \{ a _ { j } \} > 1 .\tag{SM1.12}
$$

and notice that

$$
\xi ( \tilde { \rho } _ { j } ) \le C ( a _ { \mathrm { m i n } } , \xi ) , \quad \forall j \in \mathrm { s u p p } ( \pmb { n } ) ,
$$

where the constant $C ( \boldsymbol { a } _ { \mathrm { m i n } } , \boldsymbol { \xi } )$ is increasing as $a _ { \mathrm { m i n } }$ decreases. This follows from the assumptions on $\xi ,$ which imply that it is uniformly bounded on the interval $[ a _ { \mathrm { m i n } } , \infty )$ combined with the fact that $\tilde { \rho } _ { j } \geq a _ { j } > a _ { \operatorname* { m i n } }$ for $j \in \operatorname { s u p p } ( n )$ . This implies that

$$
\xi ( \tilde { \rho } _ { j } ) \leq C ( a _ { \mathrm { m i n } } , \xi ) n _ { j } + 1 , \quad \forall j \in \mathrm { s u p p } ( n ) ,
$$

and therefore

$$
| d _ { \pmb { n } } | \leq B ( \pmb { n } ) : = \prod _ { j \in \mathrm { s u p p } ( \pmb { n } ) } \tilde { \rho } _ { j } ^ { - n _ { j } } r _ { j } ^ { n _ { j } } ( \tilde { c } n _ { j } + 1 ) ^ { \tilde { \gamma } } ,\tag{SM1.13}
$$

where $\tilde { c } = \tilde { c } ( a _ { \mathrm { m i n } } , \xi , c ) = \operatorname* { m a x } \{ C ( a _ { \mathrm { m i n } } ) , \xi ) , c \}$ is likewise increasing as $a _ { \mathrm { m i n } }$ decreases and $\tilde { \gamma } = \gamma + 1$ . We now split the product and write $B ( { \pmb n } ) = B _ { E } ( { \pmb n } ) B _ { F } ( { \pmb n } )$ and then get

$$
\| \pmb { d } \| _ { p } ^ { p } \leq \sum _ { \pmb { n } \in \mathcal { F } _ { E } } B _ { E } ( \pmb { n } ) ^ { p } \cdot \sum _ { \pmb { n } \in \mathcal { F } _ { F } } B _ { F } ( \pmb { n } ) ^ { p } = : \Sigma _ { E } \cdot \Sigma _ { F } ,
$$

where ${ \mathcal { F } } _ { S } = \{ n \in { \mathcal { F } } : \operatorname { s u p p } ( n ) \subseteq S \}$ for $S = E , F$ . To complete the proof, it sufices to show that $\Sigma _ { E } , \Sigma _ { F } < \infty$

Consider $\Sigma _ { E }$ . We have

$$
\Sigma _ { E } \leq \prod _ { j = 1 } ^ { d } \left( \sum _ { n _ { j } = 1 } ^ { \infty } ( r _ { j } / a _ { j } ) ^ { p n _ { j } } ( \tilde { c } n _ { j } + 1 ) ^ { p \tilde { \gamma } } \right) .
$$

This is finite, provided a satisfies

$$
a _ { j } > r _ { j } , \quad \forall j \in \mathbb { N } .\tag{SM1.14}
$$

Now consider $\Sigma _ { F }$ . Using the definition of ${ \tilde { \rho } } _ { j }$ , we have

$$
\begin{array} { l } { \displaystyle B _ { F } ( n ) \leq \prod _ { j \in \mathrm { s u p p } ( n _ { F } ) } \left( \frac { b _ { j } \| n _ { F } \| _ { 1 } r _ { j } } { ( 1 - s ) \varepsilon n _ { j } } \right) ^ { n _ { j } } ( \tilde { c } n _ { j } + 1 ) ^ { \tilde { \gamma } } } \\ { \leq \frac { \| n _ { F } \| _ { 1 } ! } { n _ { F } ! } \left( \frac { \mathrm { e } ( b \odot r ) _ { F } } { ( 1 - s ) \varepsilon } \right) ^ { n _ { F } } \displaystyle \prod _ { j \in \mathrm { s u p p } ( n _ { F } ) } \left( ( ( \tilde { c } + 1 ) n _ { j } ) ^ { \tilde { \gamma } } + 1 \right) } \end{array}
$$

(here $\begin{array} { r } { { \pmb n } _ { F } ! = \prod _ { j \in \mathrm { s u p p } ( { \pmb n } _ { F } ) } n _ { j } ! } \end{array}$ is the multi-index factorial). The argument is now identical to the corresponding part of the proof of [SM8, Lem. 5.5], except with b replaced by $b \odot r$ . In particular, we see that $\Sigma _ { F } < \infty$ after taking d suficiently large, provided $\pmb { b } \odot \pmb { r } \in \ell ^ { p } ( \mathbb { N } )$

To summarize, we have shown $\pmb { d } \in \ell ^ { p } ( \mathbb { N } )$ , provided there exists a sequence $\mathbf a > 1$ that satisfies (SM1.11), (SM1.12) and (SM1.14). Set

$$
a _ { j } = 1 + \operatorname* { m a x } \left\{ \frac { 1 + s } { 2 s } ( r _ { j } - 1 ) , \frac { ( 1 - s ) s \varepsilon } { 2 \| b \| _ { 1 } } \right\} .\tag{SM1.15}
$$

Since $0 < s < 1$ , we see that (SM1.12) and (SM1.14) hold. Also, letting $\mathcal { I } = \{ j \in \mathbb { N }$ $\begin{array} { r } { a _ { j } = 1 + \frac { 1 + s } { 2 s } ( r _ { j } - 1 ) \} } \end{array}$ , we see that

$$
\begin{array} { r l } { \displaystyle \sum _ { j = 1 } ^ { \infty } ( a _ { j } - 1 ) b _ { j } = \frac { 1 + s } { 2 s } \sum _ { j \in \mathcal { I } } ( r _ { j } - 1 ) b _ { j } + \displaystyle \sum _ { j \notin \mathcal { I } } \frac { ( 1 - s ) s \varepsilon } { 2 \| b \| _ { 1 } } b _ { j } } & { } \\ { \displaystyle \leq \frac { 1 + s } { 2 s } s ^ { 2 } \varepsilon + \frac { ( 1 - s ) s } { 2 } \varepsilon \varepsilon } & { } \\ { \displaystyle = s \varepsilon , } \end{array}
$$

where in the second step we used the fact that $\| \pmb { b } \odot ( \pmb { r } - \pmb { 1 } ) \| _ { 1 } \ \le \ s ^ { 2 } \varepsilon$ . Therefore (SM1.14) holds as well. This completes the proof. □

Lemma SM1.5 (Abstract anchored summability). Consider the setup of the previous lemma, except where (SM1.9) is replaced by

$$
b \odot \pmb { r } \in \ell _ { \mathsf { M } } ^ { p } ( \mathbb { N } ) , \qquad \| \pmb { b } \odot ( \pmb { r } - \mathbf { 1 } ) \| _ { 1 } < \varepsilon .
$$

Then $d \in \ell _ { \mathsf { A } } ^ { p } ( \mathcal { F } )$ and $\| d \| _ { p , \mathsf { A } } \leq C ( b , \varepsilon , r , p , c , \gamma , \xi )$

Proof. We proceed in a similar manner, but with some key diferences. Let $s \in$ $( 0 , 1 )$ be such that $\| \pmb { b } \odot ( \pmb { r } - \pmb { 1 } ) \| _ { 1 } \le s ^ { 2 } \varepsilon$ . Now define $a _ { j }$ as in (SM1.15) once more and notice that $a _ { \mathrm { m i n } } = \mathrm { m i n } \{ a _ { j } \} > 1$ . We also have

$$
\sum _ { j = 1 } ^ { \infty } ( a _ { j } - 1 ) b _ { j } \leq s \varepsilon ,
$$

as before. We now write $\mathbb { N } = E \cup F$ , where $E = [ d ]$ and $F = \mathbb { N } \backslash [ d ]$ once more and assume that $d \in \mathbb { N }$ is now suficiently large so that

$$
\sum _ { j > d } ( r _ { j } - 1 ) b _ { j } \leq \frac { ( 1 - s ) \varepsilon } { 2 \beta }
$$

for some $\beta > 0$ that will be chosen later. Given $b \in { \mathcal { F } }$ , we now let $\tilde { \pmb { \rho } } = \tilde { \pmb { \rho } } ( \pmb { b } , \beta , \varepsilon , \pmb { n } ) \geq \mathbf { 1 }$ be defined by

$$
\frac { \tilde { \rho } _ { j } + \tilde { \rho } _ { j } ^ { - 1 } } { 2 } = \left\{ \begin{array} { l l } { a _ { j } } & { j \in \mathrm { s u p p } ( n _ { E } ) } \\ { a _ { j } + \beta r _ { j } + \frac { ( 1 - s ) \varepsilon n _ { j } } { 2 b _ { j } \| n _ { F } \| _ { 1 } } } & { j \in \mathrm { s u p p } ( n _ { F } ) } \\ { 1 } & { j \notin \mathrm { s u p p } ( n ) } \end{array} \right.
$$

Observe that $\{ k : \rho _ { k } > 1 \} = \operatorname { s u p p } ( n )$ and $\tilde { \rho }$ satisfies (SM1.10). Note that (SM1.13) also holds for this choice of ${ \tilde { \rho } } .$ The rest of the proof involves constructing an upper bound $\tilde { B } ( { \pmb n } ) \geq B ( { \pmb n } )$ for which $( \tilde { B } ( \pmb { n } ) ) _ { \pmb { n } \in \mathcal { F } }$ is monotonically nonincreasing, anchored and ℓ<sup>p</sup>-summable. This will immediately imply the result.

Write $B ( \pmb { n } ) = B _ { E } ( \pmb { n } ) B _ { F } ( \pmb { n } )$ as before, and consider $B _ { E } ( { \pmb n } )$ . Let

$$
\kappa = 1 + \frac { ( 1 - s ) s \varepsilon } { 2 \| \pmb { b } \| _ { 1 } } > 1
$$

and observe that there is a constant $D = D ( \tilde { c } , \tilde { \gamma } , \kappa ) \geq 1$ such that

$$
( \widetilde c n + 1 ) ^ { \widetilde \gamma } \le D \left( \frac { 1 + \kappa } { 2 } \right) ^ { n } , \quad \forall n \in  { \mathbb { N } } _ { 0 } .
$$

Notice from (SM1.15) that

$$
\tilde { \rho } _ { j } \geq a _ { j } \geq 1 + \frac { ( 1 - s ) s \varepsilon } { 2 \| \pmb { b } \| _ { 1 } } = \kappa , \quad \forall j \in \mathrm { s u p p } ( \pmb { n } _ { E } ) .
$$

Hence

$$
B _ { E } ( \pmb { n } ) \leq D ^ { d } \prod _ { j \in \mathrm { s u p p } ( \pmb { n } _ { E } ) } \left( \frac { 1 + \kappa } { 2 \kappa } \right) ^ { n _ { j } } = : \tilde { B } _ { E } ( \pmb { n } ) .\tag{SM1.16}
$$

Now consider $B _ { F } ( { \pmb n } )$ . Let $\widehat { \pmb { t } = \pmb { b } \odot \pmb { r } }$ be a minimal monotone majorant of $b \odot r$ . Then

$$
\begin{array} { l } { \displaystyle B _ { F } ( \pmb { n } ) \leq \prod _ { j \in \mathrm { s u p p } ( { \pmb n } _ { F } ) } ( \tilde { c } n _ { j } + 1 ) ^ { \tilde { \gamma } } r _ { j } ^ { n _ { j } } \left( \beta r _ { j } + \frac { ( 1 - s ) \varepsilon n _ { j } } { 2 b _ { j } \| { \pmb n } _ { F } \| _ { 1 } } \right) ^ { - n _ { j } } } \\ { \leq \prod _ { j \in \mathrm { s u p p } ( { \pmb n } _ { F } ) } ( \tilde { c } n _ { j } + 1 ) ^ { \tilde { \gamma } } \left( \beta + \frac { ( 1 - s ) \varepsilon n _ { j } } { 2 t _ { j } \| { \pmb n } _ { F } \| _ { 1 } } \right) ^ { - n _ { j } } = : \tilde { B } _ { F } ( \pmb { n } ) } \end{array}
$$

Hence, we define $\tilde { B } ( { \pmb n } ) = \tilde { B } _ { E } ( { \pmb n } ) \tilde { B } _ { F } ( { \pmb n } )$

We now show that $\tilde { B } ( \pmb { n } )$ is monotonically nonincreasing. We do this by showing it separately for $\tilde { B } _ { E } ( \pmb { n } )$ and $\tilde { B } _ { F } ( n )$ . For the former, we see this readily holds since $\kappa > 1$ and therefore $\begin{array} { r } { \frac { 1 + \dot { \kappa } } { 2 \kappa } < 1 } \end{array}$ . Now consider $\tilde { B } _ { F } ( n )$ . To show monotonicity, we show that for every $\mathbf { \boldsymbol { n } } \in \mathcal { F }$ and every $i > d ,$ , we have $\tilde { B } _ { F } ( \pmb { n } + \pmb { e } _ { i } ) \le \tilde { B } _ { F } ( \pmb { n } )$ . Assume first that $n _ { i } \neq 0$ . Then, arguing as in the proof of [SM8, Lem. 5.2], we have

$$
\frac { \tilde { B } _ { F } ( \pmb { n } + \pmb { e } _ { i } ) } { \tilde { B } _ { F } ( \pmb { n } ) } \leq \frac { 2 ^ { \tilde { \gamma } } \mathrm { e } } { \beta } .
$$

Now suppose that $n _ { i } = 0$ . Then

$$
\frac { \tilde { B } _ { F } ( \pmb { n } + \pmb { e } _ { i } ) } { \tilde { B } _ { F } ( \pmb { n } ) } \leq \frac { ( \tilde { c } + 1 ) ^ { \tilde { \gamma } } \mathrm { e } } { \beta } .
$$

Therefore, it sufices to choose $\beta$ satisfying

(SM1.17)

$$
\beta \geq \mathrm { m a x } \left\{ 2 ^ { \tilde { \gamma } } \mathrm { e } , ( \tilde { c } + 1 ) ^ { \tilde { \gamma } } \mathrm { e } \right\} .
$$

For any such $\beta _ { ; }$ , we deduce that $\tilde { B } _ { F } ( n )$ , and therefore $\tilde { B } ( \pmb { n } )$ , is monotonically nonincreasing.

We now show that $\tilde { B } ( \pmb { n } )$ is anchored, i.e., $\tilde { B } ( e _ { j } ) \leq \tilde { B } ( e _ { i } )$ whenever $i \leq j$ . Observe that

$$
\begin{array} { r } { \tilde { B } ( e _ { j } ) = \left\{ \begin{array} { l l } { D ^ { d } \left( \frac { 1 + \kappa } { 2 \kappa } \right) } & { j \in E } \\ { ( \tilde { c } + 1 ) ^ { \tilde { \gamma } } \left( \beta + \frac { ( 1 - s ) \varepsilon } { 2 t _ { j } } \right) ^ { - 1 } } & { j \in F } \end{array} \right. } \end{array}
$$

Suppose first that $i , j \in E$ . Then $\tilde { B } ( e _ { j } ) = \tilde { B } ( e _ { i } )$ , as required. Now suppose that $i , j \in F$ . Then we require

$$
\left( \beta + \frac { ( 1 - s ) \varepsilon } { 2 t _ { j } } \right) ^ { - 1 } \leq \left( \beta + \frac { ( 1 - s ) \varepsilon } { 2 t _ { i } } \right) ^ { - 1 } .
$$

This holds, since t is a monotonically nonincreasing by assumption. Finally, if $i \in E$ $j \in F$ then we require

$$
( \widetilde c + 1 ) ^ { \widetilde \gamma } \left( \beta + \frac { ( 1 - s ) \varepsilon } { 2 t _ { j } } \right) ^ { - 1 } \leq D ^ { d } \left( \frac { 1 + \kappa } { 2 \kappa } \right) .
$$

We see that this holds, provided

$$
\beta \geq \frac { ( \tilde { c } + 1 ) ^ { \tilde { \gamma } } } { D ^ { d } \left( \frac { 1 + \kappa } { 2 \kappa } \right) } .
$$

Recall that $D \geq 1$ by assumption. Hence this condition is implied by

$$
\beta \geq ( \tilde { c } + 1 ) ^ { \tilde { \gamma } } \left( \frac { 2 \kappa } { 1 + \kappa } \right) .
$$

Combining this with (SM1.17), we now pick

$$
\beta = \operatorname* { m a x } \left\{ 2 ^ { \tilde { \gamma } } \mathrm { e } , ( \tilde { c } + 1 ) ^ { \tilde { \gamma } } \mathrm { e } , ( \tilde { c } + 1 ) ^ { \tilde { \gamma } } \left( \frac { 2 \kappa } { 1 + \kappa } \right) \right\}
$$

This concludes the third case, and therefore we have shown that $\tilde { B } ( \pmb { n } )$ is anchored. It remains to show that $d \in \ell _ { \mathsf { A } } ^ { p } ( \mathcal { F } )$ . By construction, we have

$$
\| d \| _ { p , \mathsf { A } } ^ { p } \leq \sum _ { n \in \mathcal { F } } \tilde { B } ( n ) ^ { p } = \sum _ { n \in \mathcal { F } _ { E } } \tilde { B } _ { E } ( n ) ^ { p } \cdot \sum _ { n \in \mathcal { F } _ { F } } \tilde { B } _ { F } ( n ) ^ { p } = : \tilde { \Sigma } _ { E } \cdot \tilde { \Sigma } _ { F } .
$$

Using (SM1.16), we see that

$$
\tilde { \Sigma } _ { E } = D ^ { d p } \left( \sum _ { n = 0 } ^ { \infty } \left( \frac { \kappa + 1 } { 2 \kappa } \right) ^ { n } \right) ^ { p } < \infty ,
$$

since $\kappa > 1$ . Now consider $\tilde { \Sigma } _ { F }$ . As in the previous proof, we have

$$
\tilde { B } _ { F } ( \pmb { n } ) \leq \prod _ { j \in \mathrm { s u p p } ( \pmb { n } _ { F } ) } \left( \frac { 2 t _ { j } \| \pmb { n } _ { F } \| _ { 1 } } { ( 1 - s ) \varepsilon n _ { j } } \right) ^ { n _ { j } } ( \tilde { c } n _ { j } + 1 ) ^ { \tilde { \gamma } } .
$$

We now argue in the same way as before, except with $b \odot r$ replaced by the sequence 2t, using the assumption that $\pmb { t } \in \ell ^ { p } ( \mathbb { N } )$ □

We are now ready to establish Theorems SM1.2 and SM1.3.

Proof of Theorem SM1.2. We use Lemma SM1.4. Let $\pmb { n } \in \mathcal { F }$ . Using [SM8, Lem. 5.4], we have

(SM1.18)

$$
\| c _ { \pmb { n } } \| _ { \mathcal { V } } \le \prod _ { k \in \mathrm { s u p p } ( { \pmb n } ) } \frac { \rho _ { k } ^ { - n _ { k } + 1 } } { ( \rho _ { k } - 1 ) ^ { 2 } } ( n _ { k } + 1 ) ,
$$

for any $\rho$ satisfying (SM1.10) and $\{ k : \rho _ { k } > 1 \} \supseteq \mathrm { s u p p } ( n )$ , since $f \in \mathcal { H } ( b , \varepsilon ; \mathcal { V } )$ by assumption. Now write

$$
\begin{array} { r } { \| \pmb { c } \| _ { p , \pmb { w } ; \mathcal { y } } = \| \tilde { \pmb { c } } \| _ { p ; \mathcal { y } } , } \end{array}
$$

where $\tilde { c } = ( \tilde { c } _ { n } ) _ { n \in \mathcal { F } }$ is given by $\tilde { c } _ { n } = c _ { n } w _ { n } ^ { 2 / p - 1 }$ . Using (SM1.18) and (SM1.6), we have

$$
\| \tilde { c } _ { n } \| _ { \mathcal { Y } } \leq \| f \| _ { L ^ { \infty } ( \mathcal { E } _ { \rho } ) } \prod _ { k \in \operatorname { s u p p } ( n ) } \frac { \rho _ { k } ^ { - n _ { k } + 1 } } { ( \rho _ { k } - 1 ) ^ { 2 } } ( n _ { k } + 1 ) \prod _ { k \in \operatorname { s u p p } ( n ) } ( 2 n _ { k } + 1 ) ^ { 1 / p - 1 / 2 } \zeta ( \omega _ { k } ) ^ { ( 2 / p - 1 ) n _ { k } }
$$

Therefore

$$
\| \tilde { c } _ { n } \| _ { \mathcal { V } } \leq \| f \| _ { L ^ { \infty } ( \mathcal { E } _ { \rho } ) } d _ { n } ,
$$

where

$$
d _ { n } = \prod _ { k \in \mathrm { s u p p } ( n ) } \xi ( \rho _ { k } ) \rho _ { k } ^ { - n _ { k } } r _ { p , k } ^ { n _ { k } } ( c n _ { k } + 1 ) ^ { \gamma }
$$

and

$$
\xi ( \rho ) = \frac { \rho } { ( \rho - 1 ) ^ { 2 } } , \quad r _ { p , k } = \zeta ( \omega _ { k } ) ^ { 2 / p - 1 } , \quad c = 2 , \quad \gamma = 1 / p + 1 / 2 .
$$

We now apply Lemma SM1.4 with r replaced by $\boldsymbol { r } _ { p } = ( r _ { p , k } ) _ { k \in \mathbb { N } }$ to see that $d \in \ell ^ { p } ( \mathcal { F } )$ with

$$
\begin{array} { r } { \| d \| _ { p } \leq C ( b , \varepsilon , r _ { p } , p , c , \gamma , \xi ) = C ( b , \varepsilon , \omega , p ) , } \end{array}
$$

since $\pmb { b } \odot \pmb { r } _ { p } \in \ell ^ { p } ( \mathbb { N } )$ and $\| \pmb { b } \odot ( \pmb { r _ { p } } - \pmb { 1 } ) \| _ { 1 } < \varepsilon$ by assumption. This gives the first result.

For the second result, we use Lemma SM1.1. This implies that $\sigma _ { k } ( c ) _ { q , w } ~ \leq$ $\| \pmb { c } \| _ { p , \pmb { w } ; \mathcal { y } } k ^ { 1 / q - 1 / p }$ for any $q \in ( p , 2 ]$ . Moreover, inspecting its proof, we see that the upper bound is achieved by $\| \pmb { c } - \pmb { c } _ { S } \| _ { q , w ; \mathcal { V } } ,$ , where the set $S$ satisfies $| S | _ { w } \le k$ and is chosen independently of q (and p), and depending only on c and w. □

Proof of Theorem SM1.3. Let d be the sequence defined in the previous proof. We now appeal to Lemma SM1.5 with r replaced by $\boldsymbol { r _ { p } }$ to see that $d \in \ell _ { \mathsf { A } } ^ { p } ( \mathcal { F } )$ , since $\pmb { b } \odot \pmb { r } _ { p } \in \ell _ { \mathsf { M } } ^ { p } ( \mathbb { N } )$ and $\| \pmb { b } \odot ( \pmb { r _ { p } } - \mathbf { 1 } ) \| _ { 1 } < \varepsilon$ by assumption. The result now follows.

SM1.3. Proof of Theorem 3.2. We shall prove the following generalization of Theorem 3.2, which allows for Hilbert-valued functions. For this theorem, we define the Hilbert-valued version of the distribution shift constant (3.1) for model classes of Y-valued functions $P : D  \mathcal { V }$ simply by replacing the Lebesgue norms in (3.1) by Lebesgue-Bochner norms. Given a set $S \subset { \mathcal { F } }$ , we also define the corresponding subspace of Y-valued polynomials by

$$
{ \mathcal { P } } _ { S ; \mathcal { V } } = \left\{ \sum _ { \pmb { n } \in S } c _ { \pmb { n } } \Psi _ { \pmb { n } } : c _ { \pmb { n } } \in \mathcal { V } , \ S \in \mathcal { S } \right\} .
$$

Theorem SM1.6. Let ϱ be the uniform probability measure on $[ - 1 , 1 ] ^ { \mathbb { N } } , \omega \geq { \bf 1 }$ and $D _ { \omega }$ be as in (1.3). Suppose that $f \in \mathcal { H } ( b , \varepsilon ; \mathcal { V } )$ , where b satisfies (3.3) for some $0 < p < 1$ . Then, for every k $> 0$ , there exists a set S depending on $\textstyle b , \ \varepsilon , \ \omega$ and k with $| S | \le k$ such that

$$
\Delta ( \mathcal { P } _ { S ; y } ; \varrho , \mu ) \le \sqrt { k }\tag{SM1.19}
$$

and a $g \in { \mathcal { P } } _ { S ; \mathcal { Y } }$ such that

$$
\Delta ( \mathcal { P } _ { S ; y } ; \varrho , \mu ) \| f - g \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } + \| f - g \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } \leq C ( b , \varepsilon , \omega , p ) k ^ { 1 - 1 / p }
$$

for any probability measure µ supported in $D _ { \omega }$

Proof of Theorem SM1.6. Define the weights $\pmb { u } = ( u _ { n } ) _ { n \in \mathcal { F } }$ and ${ \pmb w } = ( w _ { \pmb { n } } ) _ { \pmb { n } \in \mathcal { F } }$ where $u _ { n }$ and $w _ { n }$ are as in (SM1.5) and (SM1.6), respectively. Let $S \subset { \mathcal { F } } , | S | \leq s$ , be an index set (we will choose S later) and consider first the constant $\Delta ( \mathcal { P } _ { S ; \mathcal { V } } ; \varrho , \mu )$ . For

any $\begin{array} { r } { P = \sum _ { n \in S } c _ { n } \Psi _ { n } \in \mathcal { P } _ { S ; \mathcal { V } } } \end{array}$ , the triangle inequality, (SM1.6), the Cauchy–Schwarz inequality and Parseval’s identity give that

$$
\| P \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { \mathtt { N } } ) } \leq \sum _ { n \in S } | c _ { n } | w _ { n } \leq \sqrt { | S | _ { w } } \| P \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathtt { N } } ) } .
$$

Since $\mathcal { P } _ { S ; \mathcal { V } }$ is a linear space and $P \in \mathcal { P } _ { S ; \mathcal { V } }$ was arbitrary, it follows that

$$
\Delta ( \mathcal { P } _ { S ; \mathcal { V } } ; \varrho , \mu ) \le \sqrt { | { \cal S } | _ { w } } .
$$

Now let $P = f _ { S }$ be as in (2.3). Then, by Parseval’s identity and the definition of the $\ell _ { w } ^ { 2 } \mathrm { - n o r m }$

$$
\| f - P \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } = \| c - c _ { S } \| _ { 2 ; \mathcal { V } } = \| c - c _ { S } \| _ { 2 , w ; \mathcal { V } } ,
$$

and by the triangle inequality and (SM1.6),

$$
\| f - P \| _ { L _ { \varrho } ^ { \infty } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } \leq \sum _ { n \notin S } w _ { n } \| c _ { n } \| _ { \mathcal { V } } = \| c - c _ { S } \| _ { 1 , w ; \mathcal { V } } .
$$

Combining with (SM1.19), we deduce that

$$
\begin{array} { r } { \Delta ( \mathcal { P } _ { S ; y } ; \varrho , \mu ) \| f - P \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { Y } ) } + \| f - P \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { N } ; \mathcal { Y } ) } \leq \sqrt { | S | _ { w } } \| c - c _ { S } \| _ { 2 , w ; \mathcal { Y } } + \| c - c _ { S } \| _ { 1 , w ; \mathcal { Y } } . } \end{array}
$$

Now, the assumption (3.3) and Theorem SM1.2 imply that there is a set S with $| S | _ { w } \le k$ such that $\| c - c \| _ { q , w ; \mathcal { V } } \le C ( b , \varepsilon , \omega , p ) k ^ { 1 / q - 1 / p }$ for any $q \in ( p , 2 ]$ . We now substitute this into the previous expression using $q = 1 , 2$ to get

$$
\begin{array} { r } { \Delta ( \mathcal { P } _ { S ; \mathcal { Y } } ; \varrho , \mu ) \| f - P \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { n } ; \mathcal { Y } ) } + \| f - P \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { n } ; \mathcal { Y } ) } \le \| c \| _ { p , w ; \mathcal { Y } } k ^ { 1 - 1 / p } , \quad \mathrm { w h e r e ~ } P = f _ { S } . } \end{array}
$$

The weights ${ \pmb w } \ge { \bf 1 }$ and therefore $| S | \leq | S | _ { w } \leq k$ . The result now follows.

SM1.4. Proof of Theorem 4.1. We now prove a generalization of Theorem 4.1. The arguments for this proof are inspired by existing compressed sensing approaches to polynomial approximation of high-dimensional functions [SM30, SM29, SM6, SM4, SM8], although with substantial diferences in order to derive out-ofdistribution generalization bounds. Furthermore, we also extend the setting of Theorem 4.1 to consider the approximation of Hilbert-valued functions, as opposed to scalar-valued functions. This will be of use later when we consider operator learning.

Given $k > 0$ (its precise value will be chosen in the proof), let

$$
\Lambda = \left\{ \pmb { n } = ( n _ { j } ) _ { j = 1 } ^ { \infty } \in \mathcal { F } : \prod _ { j = 1 } ^ { \infty } ( n _ { j } + 1 ) \leq \lceil k \rceil , \ n _ { j } = 0 , \ \forall j > \lceil k \rceil \right\} .\tag{SM1.21}
$$

and define the set

$$
S = \left\{ S \subseteq \Lambda : | S | _ { w } \leq k \right\} .\tag{SM1.22}
$$

Another aspect we deal with in this section is the case where $\mathcal { V }$ is discretized to some subspace $\hat { \mathcal { V } } _ { : }$ , which we consider a Hilbert space with the same norm. This is relevant for computations – as in practice, one can never work over Y when it is

infinite dimensional – and relevant later in the context of operator learning. To this end, we now define Let $\mathcal { P } _ { S ; \hat { \mathcal { y } } } = \bigcup _ { S \in \mathcal { S } } \mathcal { P } _ { S ; \hat { \mathcal { y } } }$ . Now consider a Hilbert-valued function $f : D  \mathcal { V }$ with training data

$$
( \pmb { x } _ { i } , y _ { i } ) , \ i = 1 , \ldots , m , \quad \mathrm { w h e r e \ } y _ { i } = f ( \pmb { x } _ { i } ) + e _ { i } \in \mathcal { V }\tag{SM1.23}
$$

and $e _ { i } \in \mathcal { V }$ represents measurement noise. We now consider the Hilbert-valued nonlinear least-squares fit

$$
\operatorname* { m i n } _ { P \in \mathcal { P } _ { S ; \hat { \mathcal { V } } } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \| y _ { i } - P ( \pmb { x } _ { i } ) \| _ { \mathcal { V } } ^ { 2 } .\tag{SM1.24}
$$

The main objective of this section is to establish the following extension of Theorem 4.1 (the latter corresponds to the case $\mathcal { V } = \hat { \mathcal { V } } = \mathbb { R } )$ . For this and subsequent results, we require several additional concepts. First, we let $\hat { B } _ { y } : \mathcal { V } \to \hat { \mathcal { V } }$ be the orthogonal projection onto $\hat { \mathcal { V } }$ . Second, given an optimization problem min<sub>t</sub> $g ( t )$ and scalars $\sigma \geq 1$ and $\tau \geq 0$ , we say that t<sup>ˆ</sup> is a $( \sigma , \tau )$ -minimizer if $\begin{array} { r } { \bar { g ( t ) } \le \sigma ^ { 2 } \operatorname* { m i n } _ { t } \bar { g ( t ) } + \tau ^ { 2 } } \end{array}$

Theorem SM1.7. Let $k \geq 1 , \tau > 0 , \sigma \geq 1 , 0 < \epsilon < 1 , \omega \geq 1$ and $D _ { \omega }$ be as in (1.3). Let $f \in \mathcal { H } ( b , \varepsilon ; \mathcal { V } )$ , where b satisfies (4.2) for some $0 < p < 1$ , draw $\pmb { x } _ { 1 } , \dots , \pmb { x } _ { m } \sim _ { \mathrm { i . i . d } } \varrho _ { ; }$ where m satisfies

$$
m \geq c \cdot k \cdot \left( \log ^ { 4 } ( 2 k ) + \log ( 1 / \epsilon ) \right)
$$

for some universal constant $c > 0$ . Then, with probability at least $1 - \epsilon , a n y \ : ( \sigma , \tau ) \cdot$ minimizer <sup>ˆ</sup>f of (SM1.24) satisfies

$$
\| f - \hat { f } \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { n } ; \mathcal { Y } ) } \lesssim \frac { \sigma C ( \pmb { b } , \varepsilon , \omega , p ) } { \sqrt { \epsilon } } k ^ { 1 - 1 / p } + \frac { \sigma \sqrt { k } } { \sqrt { \epsilon } } \| f - \hat { B } _ { \mathcal { Y } } \circ f \| _ { L _ { \sigma } ^ { 2 } ( \mathbb { R } ^ { n } ; \mathcal { Y } ) } + \frac { \sigma \sqrt { k } \| e \| _ { 2 ; \mathcal { Y } } } { \sqrt { m } } + \sqrt { k } \tau _ { \eta }
$$

where $e = ( e _ { i } ) _ { i = 1 } ^ { m } \in \mathcal { V } ^ { m }$ , for any probability measure µ supported in $D _ { \omega }$

To prove this result, we require several lemmas.

Lemma SM1.8. Let $\omega \ge 1 , D _ { \omega }$ be as in (1.3), $f \in C ( D _ { \omega } ; \mathcal { V } ) , \ : x _ { 1 } , \ldots , x _ { m } \in D$ and define

$$
\alpha = \operatorname* { i n f } \left\{ \frac { \sqrt { \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \| P _ { 1 } ( { \pmb x } _ { i } ) - P _ { 2 } ( { \pmb x } _ { i } ) \| _ { \mathcal { V } } ^ { 2 } } } { \| P _ { 1 } - P _ { 2 } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } } : P _ { 1 } , P _ { 2 } \in \mathcal { P } _ { S ; \hat { \mathcal { V } } } , \ P _ { 1 } \neq P _ { 2 } \right\} .\tag{SM1.25}
$$

Suppose that $\alpha > 0$ and let <sup>ˆ</sup>f be a $( \sigma , \tau )$ -minimizer of (SM1.24). Then

$$
\begin{array} { r l } { \| f - \hat { f } \| _ { L _ { \rho } ^ { \infty } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } \leq } & { \sqrt { 2 k } \Bigg [ \| f - \hat { B } _ { \mathcal { y } } \circ f \| _ { L _ { \rho } ^ { 2 } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } + \alpha ^ { - 1 } ( \sigma + 1 ) \sqrt { \cfrac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } \| f ( { \pmb x } _ { i } ) - \hat { B } _ { \mathcal { y } } \circ f ( { \pmb x } _ { i } ) \| _ { \mathcal { y } } ^ { 2 } } } \\ & { \qquad + 2 \| f - P \| _ { L _ { \rho } ^ { 2 } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } + \alpha ^ { - 1 } ( \sigma + 1 ) \sqrt { \cfrac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } \| f ( { \pmb x } _ { i } ) - P ( { \pmb x } _ { i } ) \| _ { \mathcal { y } } ^ { 2 } } } \\ & { \qquad + \alpha ^ { - 1 } ( \sigma + 1 ) \frac { \| e \| _ { 2 ; \mathcal { V } } } { \sqrt { m } } + \alpha ^ { - 1 } \tau \Bigg ] + \| f - P \| _ { L _ { \rho } ^ { \infty } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } . } \end{array}
$$

for all $P \in \mathcal { P } _ { S ; \mathcal { V } }$ and any probability measure µ supported in $D _ { \omega }$

Notice that this error bound is with respect to polynomials $P \in \mathcal { P } _ { S ; \mathcal { y } }$ , and not $P \in \mathcal { P } _ { s ; \hat { \mathcal { y } } }$ , which is the space over which the minimization problem (SM1.24) is formulated. This is critical in establishing Theorem SM1.7.

Proof. Let $P \in \mathcal { P } _ { S ; \mathcal { y } }$ be arbitrary and observe that $\hat { B } _ { \mathcal { Y } } \circ P \in \mathcal { P } _ { S ; \hat { \mathcal { Y } } }$ . Then

$$
\begin{array} { r } { \| f - \hat { f } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } \leq \| f - \hat { B } _ { \mathcal { V } } \circ f \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } + \| \hat { B } _ { \mathcal { V } } \circ f - \hat { B } _ { \mathcal { V } } \circ P \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } + \| \hat { B } _ { \mathcal { V } } \circ P - \hat { f } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } } \end{array}
$$

For the middle term, we use the fact that $\hat { B } _ { y }$ is an orthogonal projection, and for the third term, we use the fact that $\hat { B } _ { y } \circ P , \hat { f } \in \mathcal { P } _ { S ; y }$ to get

$$
\begin{array} { r } { \| \boldsymbol { f } - \hat { \boldsymbol { f } } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathbb { N } } ; \mathcal { V } ) } \leq \| \boldsymbol { f } - \hat { \mathcal { B } } _ { \mathcal { V } } \circ \boldsymbol { f } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathbb { N } } ; \mathcal { V } ) } + \| \boldsymbol { f } - P \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathbb { N } } ; \mathcal { V } ) } } \\ { + \alpha ^ { - 1 } \sqrt { \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \| \hat { \mathcal { B } } _ { \mathcal { V } } \circ P ( \boldsymbol { x } _ { i } ) - \hat { \boldsymbol { f } } ( \boldsymbol { x } _ { i } ) \| _ { \mathcal { V } } ^ { 2 } } . } \end{array}\tag{SM1.26}
$$

Now write the training labels $y _ { i }$ in (SM1.23) as

$$
y _ { i } = { \hat { B } } _ { \mathcal { Y } } \circ f ( \pmb { x } _ { i } ) + e _ { i } ^ { \prime } , \quad \mathrm { w h e r e ~ } e _ { i } ^ { \prime } = f ( \pmb { x } _ { i } ) - { \hat { B } } _ { \mathcal { Y } } \circ f ( \pmb { x } _ { i } ) + e _ { i } .
$$

Then for the third term above, the triangle inequality and the fact that $\hat { f }$ is an approximate minimizer give

$$
\begin{array} { r l } & { \sqrt { \frac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } | \| \tilde { \mathbf { g } } _ { \boldsymbol { \Sigma } } ( \boldsymbol { \rho } ^ { \prime } ( \boldsymbol { \mathbf { x } } _ { i } ) - \tilde { \boldsymbol { f } } ( \boldsymbol { \mathbf { x } } \alpha ) | _ { s } ^ { 2 } } ) } \\ & { \leq \sqrt { \frac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } | \| \tilde { \mathbf { g } } _ { \boldsymbol { \Sigma } } - \tilde { \boldsymbol { f } } ( \boldsymbol { \mathbf { x } } _ { i } ) | _ { s } ^ { 2 } } + \sqrt { \frac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } | \| \tilde { \mathbf { g } } _ { \boldsymbol { \Sigma } } \cdot \boldsymbol { \nabla } ( \alpha _ { s } ) - \boldsymbol { \bar { f } } _ { \boldsymbol { \Sigma } } \cdot \boldsymbol { \nabla } _ { \boldsymbol { \Sigma } } \boldsymbol { \rho } ^ { \prime } ( \boldsymbol { \mathbf { x } } _ { i } ) | _ { s } ^ { 2 } } + \frac { 1 } { \sqrt { m } } \| \boldsymbol { \kappa } \| _ { 2 , s } ^ { 2 } } \\ & { \leq \sigma \sqrt { \frac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } | \| \boldsymbol { \kappa } - \frac { 1 } { m } | \boldsymbol { \kappa } - \boldsymbol { \bar { f } } ( \boldsymbol { \mathbf { x } } _ { i } ) | _ { s } ^ { 2 } } + \sqrt { \frac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } | \| \boldsymbol { \kappa } _ { \boldsymbol { \Sigma } } \cdot \boldsymbol { \nabla } ( \boldsymbol { \mathbf { x } } _ { i } ) - \boldsymbol { \bar { f } } _ { \boldsymbol { \Sigma } } \cdot \boldsymbol { \nabla } _ { \boldsymbol { \Sigma } } \boldsymbol { \rho } ^ { \prime } ( \boldsymbol { \mathbf { x } } _ { i } ) | _ { s } ^ { 2 } } + \frac { 1 } { \sqrt { m } } \| \boldsymbol { e } ^ { \prime } \| _ { s , s } + \sigma } \\ &  \leq ( - 1 ) \sqrt  \frac { 1 } { m } \displaystyle  \end{array}
$$

Here, in the penultimate step we used the fact that $\hat { B } _ { y }$ is an orthogonal projection, and in the final step we used the definition of $e ^ { \prime }$ . Substituting this into (SM1.26) now

gives that

$$
\begin{array} { r l } & { \| f - \hat { f } \| _ { L _ { \sigma } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } \leq \| f - \hat { \mathcal { B } } _ { \mathcal { y } } \circ f \| _ { L _ { \sigma } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } + \| f - P \| _ { L _ { \sigma } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } } \\ & { \qquad + \alpha ^ { - 1 } ( \sigma + 1 ) \sqrt { \cfrac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } \| P ( { \pmb x } _ { i } ) - f ( { \pmb x } _ { i } ) \| _ { \mathcal { y } } ^ { 2 } } } \\ & { \qquad + \alpha ^ { - 1 } ( \sigma + 1 ) \sqrt { \cfrac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } \| f ( { \pmb x } _ { i } ) - \hat { \mathcal { B } } _ { \mathcal { y } } \circ f ( { \pmb x } _ { i } ) \| _ { \mathcal { y } } ^ { 2 } } + \alpha ^ { - 1 } ( \sigma + 1 ) \frac { \| e \| _ { 2 ; \mathcal { y } } } { \sqrt { m } } + \alpha ^ { - 1 } \tau . } \end{array}
$$

Now observe that $\hat { f } - P \in \mathcal { P } _ { S ^ { \prime } ; \mathcal { V } }$ for some set $S ^ { \prime }$ with $| S ^ { \prime } | _ { w } \leq 2 k$ . Using (SM1.19) with $S ^ { \prime }$ we get $\Delta ( \mathcal { P } _ { S ^ { \prime } ; \mathcal { V } } ; \varrho , \mu ) \le \sqrt { 2 k }$ . We now apply (3.2) (or, more precisely, the Hilbert-valued extension of this bound, which is proved in exactly the same way) to deduce that

$$
\begin{array} { r } { \| f - \hat { f } \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } \leq \sqrt { 2 k } \left( \| f - P \| _ { L _ { \rho } ^ { 2 } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } + \| f - \hat { f } \| _ { L _ { \rho } ^ { 2 } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } \right) + \| f - P \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } . } \end{array}
$$

Combining this with the previous estimate now gives the result.

Lemma SM1.9. Let $0 < \epsilon < 1 , \pmb { x } _ { 1 } , \dotsc , \pmb { x } _ { m } \sim _ { \mathrm { i . i . d . } } \ \varrho$ and consider the constant α defined in (SM1.25). Suppose that $k \geq 1$ . Then $\mathbb { P } ( \alpha < 1 / 2 ) \le \epsilon$ , provided

$$
m \geq c \cdot k \cdot \left( \log ^ { 4 } ( 2 k ) + \log ( 1 / \epsilon ) \right) ,\tag{SM1.27}
$$

where $c > 0$ is a universal constant.

Proof. Let $P _ { 1 } , P _ { 2 } \in \mathcal { P } _ { S ; y }$ be arbitrary and observe that $P _ { 1 } - P _ { 2 }$ can be expressed as $\begin{array} { r } { P _ { 1 } - P _ { 2 } = \sum _ { n \in S ^ { \prime } } c _ { n } \Psi _ { n } } \end{array}$ where $S ^ { \prime } \subseteq \Lambda , | S ^ { \prime } | _ { w } \leq 2 k$ . Define $c _ { n } = 0 \in \mathcal { V }$ for $\pm \nobreakspace S ^ { \prime }$ Now $\{ \psi _ { j } \} _ { j }$ be an orthonormal basis of $\mathcal { V }$ and write $\begin{array} { r } { c _ { n } = \sum _ { j } d _ { n , j } \psi _ { j } } \end{array}$ for $d _ { n , j } \in \mathbb { R }$ . Let $\pmb { d } _ { j } = ( d _ { n , j } ) _ { n \in \Lambda }$ . Then, by Parseval’s identity,

$$
\left\| P _ { 1 } - P _ { 2 } \right\| _ { L ^ { 2 } ( { \mathbb { R } ^ { n } } ; \mathcal { y } ) } ^ { 2 } = \sum _ { n \in \Lambda } \left\| c _ { n } \right\| _ { \mathcal { y } } ^ { 2 } = \sum _ { n \in \Lambda } \sum _ { j } | d _ { n , j } | ^ { 2 } = \sum _ { j } \| d _ { j } \| _ { 2 } ^ { 2 } ,
$$

and

$$
\frac { 1 } { m } \sum _ { i = 1 } ^ { m } \| P _ { 1 } ( \pmb { x } _ { i } ) - P _ { 2 } ( \pmb { x } _ { i } ) \| _ { \mathcal { V } } ^ { 2 } = \sum _ { j } \| \pmb { A } \pmb { d } _ { j } \| _ { 2 } ^ { 2 } ,
$$

where $\pmb { A } \in \mathbb { R } ^ { m \times N }$ is the matrix $\begin{array} { r } { A = \frac { 1 } { \sqrt { m } } \left( \Psi _ { n } ( \pmb { x } _ { i } ) \right) _ { i \in [ m ] , \pmb { n } \in \Lambda } . } \end{array}$ . We deduce that $\alpha \geq \operatorname* { i n f } _ { z \in T } \| A z \| _ { 2 } ,$ where $T = \left\{ z = ( z _ { n } ) _ { n \in \Lambda } \in \mathbb { R } ^ { N } , \ \| z \| _ { 2 } = 1 , \ | \mathrm { s u p p } ( z ) | _ { w } \leq 2 k \right\}$

Notice that $T \subseteq \{ z : \| z \| _ { 1 . w } \leq \sqrt { 2 k } \}$ by the Cauchy–Schwarz inequality. Now define the random vector $\pmb { X } \in \mathbb { R } ^ { N }$ by $\pmb { X } = ( \Psi _ { \pmb { n } } ( \pmb { x } ) ) _ { \pmb { n } \in \Lambda }$ for $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } } \sim \varrho \mathbf { \Delta } _ { \mathbf { \lambda } }$ and observe that

$$
\mathbb { E } | \langle X , z \rangle | ^ { 2 } = \| z \| _ { 2 } ^ { 2 }\tag{SM1.28}
$$

by Parseval’s identity, and also that

$$
\left. \left. X , e _ { n } \right. \right. \leq \left. \Psi _ { n } \right. _ { L _ { \rho } ^ { \infty } ( \mathbb { R } ^ { \mathbb { N } } ) } = u _ { n } \leq w _ { n } , \quad \forall n \in \Lambda ,
$$

almost surely, due to (SM1.5) and (SM1.6). We now apply [SM11, Thm. 2.13]. This gives that there are univeral constants $\kappa , c _ { 0 } , c _ { 1 } > 0$ such that, if

$$
m \ge c _ { 0 } \cdot \delta ^ { - 2 } \cdot k \cdot \log ( \mathrm { e } | \Lambda | ) \cdot \log ^ { 2 } ( k / \delta ) \log ^ { 2 } ( 1 / \delta )\tag{SM1.29}
$$

for some $\delta \in ( 0 , \kappa )$ , then

$$
\operatorname* { s u p } _ { z \in T } \left| \frac { 1 } { m } \sum _ { i = 1 } ^ { m } | \langle X _ { i } , z \rangle | ^ { 2 } - \mathbb { E } | \langle X , z \rangle | ^ { 2 } \right| \leq c _ { 1 } \delta \left( 1 + \operatorname* { s u p } _ { z \in T } \mathbb { E } | \langle X , z \rangle | ^ { 2 } \right)
$$

with probability at least $1 - 2 \exp ( - c _ { 2 } \delta ^ { 2 } m / k )$ , where $X _ { 1 } , \ldots , X _ { m }$ are independent copies of X. Notice that $\begin{array} { r } { \frac { 1 } { m } \sum _ { i = 1 } ^ { m } | \langle X _ { i } , z \rangle | ^ { 2 } = \| A z \| _ { 2 } ^ { 2 } . } \end{array}$ . Using this, (SM1.28) and rearranging, we see that

$$
\alpha ^ { 2 } \geq \operatorname* { i n f } _ { z \in T } \left\| A z \right\| _ { 2 } ^ { 2 } \geq 1 - 2 c _ { 1 } \delta ,
$$

with the same probability. Set $\delta = \operatorname* { m i n } \{ \kappa / 2 , 3 / ( 8 c _ { 1 } ) \}$ . Then we see that

$$
\mathbb { P } ( \alpha < 1 / 2 ) \le 2 \exp ( - c _ { 2 } \delta ^ { 2 } m / k ) ,
$$

provided (SM1.29) holds with this value of $\delta .$ . Therefore $\mathbb { P } ( \alpha < 1 / 2 ) < \epsilon$ , provided (SM1.29) holds, along with

$$
m \geq c _ { 2 } ^ { - 1 } \cdot \delta ^ { - 2 } \cdot k \cdot \log ( 2 / \epsilon )
$$

for the given value of $\delta .$ . In particular, it sufices for

$$
m \geq c _ { 3 } \cdot k \cdot \left( \log ( \operatorname { e } | \Lambda | ) \cdot \log ^ { 2 } ( 2 k ) + \log ( 1 / \epsilon ) \right) ,\tag{SM1.30}
$$

provided $k \geq 1 . \mathrm { ~ A ~ }$ standard bound (see, e.g., the proof of [SM5, Lem. 6.4]) gives that $\log ( E | \Lambda | ) \leq 4 \log ^ { 2 } ( \mathrm { e } \lceil k \rceil ) \leq 4 \log ^ { 2 } ( 2 \mathrm { e } k )$ . Hence (SM1.30) is implied by

$$
m \geq c _ { 4 } \cdot k \cdot \left( \log ^ { 4 } ( 2 k ) + \log ( 1 / \epsilon ) \right) .\tag{SM1.31}
$$

This complete the proof.

We are now ready to establish Theorem SM1.7.

Proof of Theorem SM1.7. For a suitable choice of the universal constant c we see that (SM1.27) holds with ϵ replaced by $\epsilon / 3$ . Hence Lemma SM1.9 gives that $\mathbb { P } ( \alpha < 1 / 2 ) \le \epsilon / 3$ . We now apply Lemma SM1.8 to deduce that

$$
\begin{array} { r l } { \| f - \hat { f } \| _ { L _ { \mu } ^ { \infty } ( { \mathbb { R } ^ { n } ; { \mathcal { Y } } } ) } \lesssim } & { \sqrt { k } \Bigg [ \| f - \hat { B } _ { \mathcal { Y } } \circ f \| _ { L _ { \sigma } ^ { 2 } ( { \mathbb { R } ^ { n } ; { \mathcal { Y } } } ) } + \sigma \sqrt { \cfrac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } \| f ( { \pmb x } _ { i } ) - \hat { B } _ { \mathcal { Y } } \circ f ( { \pmb x } _ { i } ) \| _ { \mathcal { Y } } ^ { 2 } } } \\ & { \qquad + \| f - P \| _ { L _ { \sigma } ^ { 2 } ( { \mathbb { R } ^ { n } ; { \mathcal { Y } } } ) } + \sigma \sqrt { \cfrac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } \| f ( { \pmb x } _ { i } ) - P ( { \pmb x } _ { i } ) \| _ { \mathcal { Y } } ^ { 2 } } + \sigma \frac { \| e \| _ { 2 ; \mathcal { Y } } } { \sqrt { m } } + \tau \Bigg ] } \\ & { \qquad + \| f - P \| _ { L _ { \mu } ^ { \infty } ( { \mathbb { R } ^ { n } ; { \mathcal { Y } } } ) } } \end{array}\tag{SM1.32}
$$

for all $P \in \mathcal { P } _ { S }$ , with probability at least $1 - \epsilon / 3$

We next fix a suitable P. Let $S \subset \mathcal { F } , | S | _ { w } \leq k$ , be the set specified in Theorem SM1.2 (notice that (SM1.7) holds by assumption). Now write $S = S _ { 1 } \cup S _ { 2 }$ , where $S _ { 1 } = S \cap \Lambda$ and $S _ { 2 } = S \cap \Lambda ^ { c }$ . Notice that $| S _ { i } | _ { w } \leq k , i = 1 , 2$ , and therefore $S _ { 1 } \in S$ Finally, let $P = f _ { S _ { 1 } }$ . Now consider the middle term of (SM1.32). Let X be the random variable $\begin{array} { r } { \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \| P ( \pmb { x } _ { i } ) - f ( \pmb { x } _ { i } ) \| _ { \mathcal { V } } ^ { 2 } } \end{array}$ and notice that $\operatorname { \mathbb { E } } ( X ) = \left\| f - P \right\| _ { L _ { \varrho } ^ { 2 } ( { \mathbb { R } ^ { N } } ; { \mathscr { y } } ) } ^ { 2 } .$ Hence Markov’s inequality gives that

$$
\mathbb { P } \left( \sqrt { \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left\| P ( { \boldsymbol x } _ { i } ) - f ( { \boldsymbol x } _ { i } ) \right\| _ { \mathcal { Y } } ^ { 2 } } \ge \frac { \| f - P \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { n } ; \mathcal { Y } ) } } { \sqrt { \epsilon / 3 } } \right) \le \epsilon / 3 .
$$

Similarly, Markov’s inequality also gives that

$$
\mathbb { P } \left( { \sqrt { \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left\| f ( \pmb { x } _ { i } ) - { \hat { B } } _ { \mathcal { Y } } \circ f ( \pmb { x } _ { i } ) \right\| _ { \mathcal { Y } } ^ { 2 } } } \geq \frac { \left\| f - { \hat { B } } _ { \mathcal { Y } } \circ f \right\| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { Y } ) } } { \sqrt { \epsilon / 3 } } \right) \leq \epsilon / 3 .
$$

Using these two estimates, (SM1.32) and the union bound, we deduce that

(SM1.33)

$$
\begin{array} { r } { \| f - \hat { f } \| _ { L _ { \rho } ^ { \infty } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } \lesssim \sqrt { k } \left( \frac { \sigma } { \sqrt { \epsilon } } \| f - \hat { B } _ { \mathcal { V } } \circ f \| _ { L _ { \rho } ^ { 2 } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } + \frac { \sigma } { \sqrt { \epsilon } } \| f - f _ { S _ { 1 } } \| _ { L _ { \rho } ^ { 2 } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } + \sigma \frac { \| e \| _ { 2 ; \mathcal { V } } } { \sqrt { m } } + \tau \right) } \\ { + \| f - f _ { S _ { 1 } } \| _ { L _ { \rho } ^ { \mu } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } } \end{array}
$$

with probability at least $1 - \epsilon .$

Finally, we bound the two terms involving $f - f _ { S _ { 1 } }$ in (SM1.33). Let $\pmb { c } = ( c _ { n } ) _ { n \in \mathcal { F } }$ be the coeficients of $f .$ Since $S _ { 1 } = S \cap \Lambda$ , we have

$$
\begin{array} { r l } & { \| f - f _ { S _ { 1 } } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } = \| c - c _ { S _ { 1 } } \| _ { 2 ; \mathcal { V } } } \\ & { \qquad \leq \| c - c _ { S } \| _ { 2 ; \mathcal { V } } + \| c - c _ { \Lambda } \| _ { 2 ; \mathcal { V } } } \\ & { \qquad = \| c - c _ { S } \| _ { 2 , w ; \mathcal { V } } + \| c - c _ { \Lambda } \| _ { 2 , w ; \mathcal { V } } , } \end{array}
$$

where ${ \pmb w } = ( w _ { \pmb { n } } ) _ { \pmb { n } \in \mathcal { F } }$ is as in (SM1.6) and

$$
\begin{array} { r } { \| f - f _ { S _ { 1 } } \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } = \| c - c _ { S _ { 1 } } \| _ { 1 , w } \leq \| c - c _ { S } \| _ { 1 , w ; \mathcal { V } } + \| c - c _ { \Lambda } \| _ { 1 , w ; \mathcal { V } } . } \end{array}
$$

By definition of S, Theorem SM1.2 gives that

$$
\| c - c _ { S } \| _ { q , w ; \mathcal { V } } \le C ( b , \varepsilon , \omega , p ) k ^ { 1 / q - 1 / p }
$$

for all $q \in ( p , 2 ]$ . Theorem SM1.3 (notice that (SM1.8) holds by assumption) implies that there exists an anchored set $T$ with $| T | \leq | k |$ such that

$$
\lVert c - \pmb { c } _ { T } \rVert _ { q , w ; \mathcal { V } } \leq C ( \pmb { b } , \varepsilon , \omega , p ) \lceil k \rceil ^ { 1 / q - 1 / p }
$$

for all $q \in ( p , 2 ]$ . Moreover, Λ contains all anchored sets of size at most n [SM6, Prop. 2.18]. Hence $T \subseteq \Lambda$ and we deduce tha

$$
\begin{array} { r } { \| \pmb { c } - \pmb { c } _ { \Lambda } \| _ { q , w ; \mathcal { V } } \le C ( \pmb { b } , \varepsilon , \omega , p ) \lceil k \rceil ^ { 1 / q - 1 / p } } \end{array}
$$

for all $q \in ( p , 2 ]$ . Combining the various estimates, we conclude that

$$
\begin{array} { r } { \| f - f _ { S _ { 1 } } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } \leq C ( \boldsymbol { b } , \boldsymbol { \varepsilon } , \boldsymbol { \omega } , \boldsymbol { p } ) k ^ { 1 / 2 - 1 / p } , \qquad \| f - f _ { S _ { 1 } } \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } \leq C ( \boldsymbol { b } , \boldsymbol { \varepsilon } , \boldsymbol { \omega } , \boldsymbol { p } ) k ^ { 1 - 1 / p } . } \end{array}
$$

Substituting this into (SM1.33) we get

$$
\| f - \hat { f } \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } \leq \frac { \sigma } { \sqrt { \epsilon } } C ( b , \varepsilon , \omega , p ) k ^ { 1 - 1 / p } + \frac { \sigma \sqrt { k } } { \sqrt { \epsilon } } \| f - \hat { B } _ { \mathcal { V } } \circ f \| _ { L _ { \sigma } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } + \frac { \sigma \sqrt { k } \| e \| _ { 2 } } { \sqrt { m } } + \sqrt { k } \tau .\tag{SM1.34}
$$

with probability at least $1 - \epsilon ,$ as required.

Proof of Theorem 4.1. Let c be the constant from Lemma SM1.9 and assume, without loss of generality, that $c \geq 1 / \log ^ { 4 } ( 2 )$ . Set

$$
k = \frac { m } { c \left( \log ^ { 4 } ( m ) + \log ( 1 / \epsilon ) \right) } .\tag{SM1.35}
$$

Notice that $k \ \leq \ m / ( c \log ^ { 4 } ( 2 ) )$ whenever $m \geq 2$ and therefore $k \ \leq \ m$ . Now let $\bar { m } = \bar { m } ( \epsilon ) \in \mathbb { N }$ be the smallest $m \in \mathbb { N } , m \geq 2$ , such that $k \geq 1$ . Let $\mathcal { V } = \hat { \mathcal { V } } = \mathbb { R }$ with the Euclidean inner product, so that, in particular, $\hat { B } _ { y } = \mathcal { I } _ { y }$ is the identity. We now apply Theorem SM1.7 with this value of $k , \sigma = 1$ and $\tau = 0$ . This yields the result.

SM1.5. Proofs of Theorems 4.4 and 5.6. We now prove Theorems 4.4 and 5.6. Our main focus in the proof of Theorem 5.6, since Theorem 4.4 will then follow as a special case.

Our broad approach is based on well-established technique of emulation of polynomials via DNNs, or in this case, DNOs, [SM5, SM18, SM20, SM21, SM24, SM33, SM31, SM25, SM28, SM32] (see [SM3, §7.1] for a review). As noted earlier, emulation techniques are normally used to assert the existence of DNNs/DNOs with certain approximation guarantees. Conversely, we follow ideas developed in [SM2,SM5,SM7] that emulate a whole polynomial training problem as a DNO training problem, thus showing the existence of DL strategies with guaranteed error bounds.

Our proof proceeds in three steps. These steps are broadly similar to those used in [SM7], which develops generalization bounds learning holomorphic operators, but considers only the in-distribution setting. Our results generalize these by considering arbitrary test distributions. First, we introduce the polynomial training problem

$$
\operatorname* { m i n } _ { P \in \mathcal { P } _ { { \mathcal { S } } ; \hat { \mathcal { V } } } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \| Y _ { i } - P \circ \hat { \mathcal { E } } _ { \mathcal { X } } ( X _ { i } ) \| _ { \mathcal { V } } ^ { 2 } ,\tag{SM1.36}
$$

where $s$ is as defined in (SM1.22) with Λ as in (SM1.21). The second step is to then emulate the relevant polynomials in the hypothesis class a DNOs N (§SM1.5.1). Next, we show that the weights corresponding to an approximate minimizer of the associated DNO training problem yield coeficients of a polynomial that is an approximate minimizer of (SM1.36) (§SM1.5.2). Using this, we establish a general out-of-distribution error bound for the DNO training problem, based on the analogous bound for the polynomial training problem. Finally, we specialize this to the settings of Theorems 4.4 and 5.6 to conclude the proof (§SM1.5.3).

SM1.5.1. Construction of N. Let $0 < k \leq d _ { \mathcal { X } }$ be a number whose value will be chosen later. We first require the following lemma.

Lemma SM1.10. Let $\Gamma \subseteq \Lambda$ and $m ( \Gamma ) = \operatorname* { m a x } _ { n \in \Gamma } \left\| n \right\| _ { 1 } < \infty$ . Then there exists a fully-connected family $\mathcal { N } _ { o }$ of tanh $D N N s \ \mathbb { R } ^ { \lceil k \rceil } \to \mathbb { R }$ with

$$
\mathrm { w i d t h } ( { \mathcal N } _ { o } ) \lesssim m ( \Gamma ) , \quad \mathrm { d e p t h } ( { \mathcal N } _ { o } ) \lesssim \log ( m ( \Gamma ) ) ,
$$

such that, for any $0 < \delta < 1 , \omega \geq { \bf 1 }$ and $\pmb { n } \in \Gamma$ , there is a $N _ { n } \in \mathcal N _ { o }$ satisfying

$$
\operatorname* { s u p } _ { \pmb { x } \in D _ { \omega } } | N _ { \pmb { n } } ( \pmb { x } ) - \Psi _ { \pmb { n } } ( \pmb { x } ) | \leq \delta .
$$

Moreover, the zero network $0 : x \mapsto 0$ also belongs to $\mathcal { N } _ { o }$ (trivially, since tanh $( 0 ) = 0 )$

Note here we slightly abuse notation and consider the DNNs $N _ { n }$ both as maps with domain $\mathbb { R } ^ { \lceil k \rceil }$ and maps with domain $\mathbb { R } ^ { \mathbb { N } }$ that depend only on the first ⌈k⌉ entries of the given input. This is permissible, since the Legendre polynomials $\Psi _ { n } , \pmb { n } \in \Lambda$ depend on the first $\lceil k \rceil$ variables only. This lemma is based on [SM7, Lem. D.9], which is itself based on [SM18, SM28]. Notice that [SM7, Lem. D.9] considers only the domain $D = D _ { 1 }$ . However, the proof extends to $\omega \geq 1$ . In particular, the width and depth bounds do not depend on $\omega$ (although the weights and biases of $N _ { n } \ \mathrm { d o } )$ This is a particular feature of tanh $\mathrm { D N N s }$ , which makes them particularly desirable over, for instance, ReLU DNNs for out-of-distribution generalization.

Now fix $\delta > 0$ (its value will be chosen later in the proof), let $\Gamma = \cup _ { S \in { S } } S$ and $\mathcal { N } _ { o }$ and $N _ { n } , n \in \Gamma$ , be as in the above lemma. We now estimate $m ( \Gamma )$ . Let $\mathbf { \pmb { n } } \in \Gamma$ . Since ${ \boldsymbol { n } } \in S$ for some $S \in S$ and, by construction, $| S | _ { w } \le k$ , we have

$$
\prod _ { j = 1 } ^ { \infty } ( 2 n _ { j } + 1 ) \zeta ( \omega _ { j } ) ^ { 2 n _ { j } } = w _ { n } ^ { 2 } \leq k .
$$

Since $\zeta ( \omega ) = \omega + \sqrt { \omega ^ { 2 } - 1 } \ge \omega$ , it follows that

$$
\begin{array} { r } { \| \pmb { n } \| _ { 1 } \exp ( 2 \| \pmb { n } \| _ { 1 } \log ( \omega _ { \mathrm { m i n } } ) ) \le k , } \end{array}
$$

where $\omega _ { \mathrm { m i n } } = \mathrm { m i n } \{ \omega _ { j } \} > 1$ by assumption, and therefore

$$
m ( \Gamma ) \lesssim \frac { \log ( k ) } { \log ( \omega _ { \mathrm { m i n } } ) } .
$$

Notice that $| S | \le | S | _ { w } \le k$ for any $S \in S$ . We now define the family $\mathcal { N }$ of DNNs $N : \mathbb { R } ^ { d _ { \mathcal { X } } }  \dot { \mathbb { R } } ^ { d _ { \mathcal { Y } } }$ as

$$
\mathcal { N } = \left\{ N = C \left[ \begin{array} { c } { N _ { n _ { 1 } } } \\ { \vdots } \\ { N _ { n _ { | S | } } } \\ { 0 } \\ { \vdots } \\ { 0 } \end{array} \right] : S = \{ n _ { 1 } , \ldots , n _ { | S | } \} \in \mathcal { S } , \ C \in \mathbb { R } ^ { d _ { \mathcal { V } } \times \lceil k \rceil } \right\} .
$$

Here 0 denotes the zero network. As before, we slightly abuse notation by allowing N to be a function whose domain is either $\mathbb { R } ^ { \lceil k \rceil } , \mathbb { R } ^ { d _ { x } }$ or $\mathbb { R } ^ { \infty }$ depending on the context. Notice that this family satisfies

$$
\mathrm { w i d t h } ( { \mathcal { N } } ) \leq \mathrm { w i d t h } ( { \mathcal { N } } _ { o } ) \operatorname* { m a x } _ { S \in S } | S | \lesssim k m ( \Gamma ) \lesssim { \frac { k \log ( k ) } { \log ( \omega _ { \operatorname* { m i n } } ) } }\tag{SM1.37}
$$

and

$$
\mathrm { d e p t h } ( \mathcal { N } ) \leq \mathrm { d e p t h } ( \mathcal { N } _ { o } ) \lesssim \log ( m ( \Gamma ) ) \lesssim \log \left( \frac { \log ( k ) } { \log ( \omega _ { \operatorname* { m i n } } ) } \right) .\tag{SM1.38}
$$

Finally, given $N \in \mathcal N$ and its associated matrix $C \in \mathbb { R } ^ { d _ { \mathcal { y } } \times \lceil k \rceil }$ . Let $c _ { 1 } , \ldots , c _ { \lceil k \rceil }$ denote the columns of $^ { C , }$ so that $N$ can be expressed as

$$
N = \sum _ { i = 1 } ^ { | S | } c _ { i } N _ { { n } _ { i } } .
$$

Moreover, we have

$$
\hat { \mathcal { D } } _ { \mathcal { Y } } \circ N = \sum _ { i = 1 } ^ { | S | } \hat { \mathcal { D } } _ { \mathcal { Y } } ( \boldsymbol { c } _ { i } ) N _ { n _ { i } } = \sum _ { i = 1 } ^ { | S | } \boldsymbol { c } _ { i } N _ { n _ { i } } , \quad \mathrm { w h e r e ~ } \boldsymbol { c } _ { i } \in \hat { \mathcal { V } } ,
$$

where $\mathcal { \hat { y } } = \mathcal { \hat { D } } _ { \mathcal { y } } ( \mathbb { R } ^ { d y } ) \subseteq \mathcal { y }$ . Therefore, we can associate $\mathcal { N }$ with the space of functions $\mathcal { Q } _ { S ; \hat { \mathcal { V } } }$ given by

$$
\mathcal { Q } _ { S ; \hat { \mathcal { V } } } = \left\{ \sum _ { n \in S } c _ { n } N _ { n } : c _ { n } \in \hat { \mathcal { V } } , \ S \in \mathcal { S } \right\} .
$$

SM1.5.2. Polynomial training problem and approximate DNN minimizers. We first require the following lemma.

Lemma SM1.11. Let S, $\{ N _ { n } \} _ { n \in \Gamma }$ be as defined above and $( { \mathcal { Z } } , \| \cdot \| _ { \mathcal { Z } } )$ be a Hilbert space. Let $S \in S$ and suppose that

$$
p = \sum _ { n \in S } c _ { n } \Psi _ { n } , \qquad \tilde { p } = \sum _ { n \in S } c _ { n } N _ { n } ,
$$

where $c _ { n } \in { \mathcal { Z } }$ . Then

$$
\operatorname* { s u p } _ { \pmb { x } \in D _ { \omega } } \| p ( \pmb { x } ) - \widetilde { p } ( \pmb { x } ) \| _ { \mathcal { Z } } \leq \delta \sqrt { | S | } \| p \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathbb { N } } ; \mathcal { Z } ) } .
$$

Proof. We have

$$
\| p ( { \pmb x } ) - \widetilde p ( { \pmb x } ) \| _ { \mathcal { Z } } \le \sum _ { n \in S } \| c _ { n } \| _ { \mathcal { Z } } | \Psi _ { n } ( { \pmb x } ) - N _ { n } ( { \pmb x } ) | \le \delta \sum _ { n \in S } \| c _ { n } \| _ { \mathcal { Z } } .
$$

The result now follows from the Cauchy–Schwarz inequality and Parseval’s identity. □

We now show that approximate minimizers of the DNO training problem yield approximate minimizers of the polynomial training problem.

Lemma SM1.12. Suppose that $\alpha > 0$ , where α is as in (SM1.25) with Y replaced by $\hat { \mathcal { V } }$ , let $\mathcal { N }$ be the DNN model class defined above and $\hat { N }$ be any $( \sigma , \tau )$ -approximate minimizer of (5.3). Write

$$
\mathcal { D } y \circ \hat { N } = \sum _ { n \in S } \hat { c } _ { n } N _ { n } \in \mathcal { Q } _ { \mathcal { S } ; \hat { \mathcal { V } } }
$$

where $S \in S$ and $\hat { c } _ { n } \in \hat { \mathcal { V } }$ , and define the polynomial

$$
\hat { P } = \sum _ { n \in S } \hat { c } _ { n } \Psi _ { n } \in \mathcal { P } _ { S ; \hat { \mathcal { V } } } .
$$

Then $\hat { P }$ is a $( \sigma ^ { \prime } , \tau ^ { \prime } )$ -approximate minimizer of (SM1.36), where

$$
\begin{array} { l } { \displaystyle \sigma ^ { \prime } \leq \sigma \big ( 1 + \delta \sqrt { k } / \alpha \big ) } \\ { \displaystyle \tau ^ { \prime } \leq \tau + \frac { \sigma \delta \sqrt { k } } { \alpha } \left( \| F \| _ { L _ { v } ^ { \infty } ( \mathcal { X } ; \mathcal { Y } ) } + \frac { 1 } { \sqrt { m } } \| E \| _ { 2 ; \mathcal { Y } } \right) + \delta \sqrt { k } \| \hat { P } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathbb { N } } ; \mathcal { Y } ) } . } \end{array}\tag{SM1.39}
$$

Proof. Let $\textstyle P = \sum _ { n \in S } c _ { n } \Psi _ { n } \ \in \ P _ { S ; \hat { \mathcal { V } } }$ be arbitrary and $N \in \mathcal N$ be such that $\begin{array} { r } { \hat { \mathcal { D } } _ { \mathcal { Y } } \circ N = \sum _ { n \in S } c _ { n } N _ { n } } \end{array}$ . Then

$$
\begin{array} { r l } { \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | x _ { i } - \hat { r } _ { i } \phi _ { i } \hat { r } _ { i } \hat { r } _ { i } \hat { r } _ { i } \hat { r } _ { j } \hat { r } _ { j } } \le \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | x _ { i } - \hat { r } _ { i } \phi _ { i } \hat { r } _ { i } \hat { r } _ { j } \hat { r } _ { j } \hat { r } _ { j } \hat { r } _ { j } \hat { r } _ { j } \hat { r } _ { j } \hat { } _ { i } \hat { r } _ { j } } \le \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | x _ { i } - \hat { r } _ { i } \phi _ { i } \hat { r } _ { i } \hat { r } _ { j } \hat { r } _ { j } \hat { r } _ { j } \hat { } _ { i } \hat { } _ { i } \hat { r } _ { j } \hat { } _ { i } } \ }  & { } \\ { + \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | x _ { i } - \hat { r } _ { i } \phi _ { i } \hat { r } _ { i } \hat { r } _ { j } \hat { r } _ { j } \hat { r } _ { j } \hat { } _ { i } \hat { r } _ { j } \hat { } _ { i } \hat { } _ { i } \hat { } _ { } \hat { } \hat { } { } \hat { } \xi _ { } } } & { } \\ { \le \le \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | x _ { i } - \hat { r } _ { i } \phi _ { i } \hat { r } _ { j } \hat { r } _ { j } \hat { r } _ { i } \hat { r } _ { j } \hat { } _ { i } \hat { } _ { } \xi _ { } } \ } & { \ } \\  + \sqrt  \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | x _ { i } - \hat { r } _ { i } \phi _ { i } \hat { r } _ { j } \hat  \end{array}
$$

and therefore

$$
\begin{array} { r l } & { \sqrt { \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left\| Y _ { i } - \hat { P } \circ \hat { \mathcal { E } } _ { \mathcal { X } } ( X _ { i } ) \right\| _ { \mathcal { Y } } ^ { 2 } } \leq \sigma \sqrt { \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left\| Y _ { i } - P \circ \hat { \mathcal { E } } _ { \mathcal { X } } ( X _ { i } ) \right\| _ { \mathcal { Y } } ^ { 2 } } + \tau } \\ & { \qquad + \left\| \hat { P } - \hat { \mathcal { D } } _ { \mathcal { Y } } \circ \hat { N } \right\| _ { L _ { \mu } ^ { \infty } ( { \mathbb R } ^ { N } ; { \mathcal { Y } } ) } + \sigma \| P - \hat { \mathcal { D } } _ { \mathcal { Y } } \circ N \| _ { L _ { \mu } ^ { \infty } ( { \mathbb R } ^ { N } ; { \mathcal { Y } } ) } , } \end{array}
$$

where we recall that $\hat { \mathcal { E } } _ { \mathcal { X } } ( X _ { i } ) \in D$ by Assumption 5.1(ii). We now apply Lemma

SM1.11 to the latter two terms to get, after recalling that $| S | \le k$ since $S \in S$

$$
\begin{array} { r } { \sqrt { \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left\| Y _ { i } - \hat { P } \circ \hat { \mathcal { E } } _ { \mathcal { X } } ( X _ { i } ) \right\| _ { \mathcal { Y } } ^ { 2 } } \leq \sigma \sqrt { \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left\| Y _ { i } - P \circ \hat { \mathcal { E } } _ { \mathcal { X } } ( X _ { i } ) \right\| _ { \mathcal { Y } } ^ { 2 } } + \tau } \\ { + \delta \sqrt { k } \| \hat { P } \| _ { L _ { \sigma } ^ { 2 } ( { \mathbb { R } ^ { n } } ; \mathcal { Y } ) } + \sigma \delta \sqrt { k } \| P \| _ { L _ { \sigma } ^ { 2 } ( { \mathbb { R } ^ { n } } ; \mathcal { Y } ) } . } \end{array}
$$

It remains to estimate the final term. Applying (SM1.25) with $P _ { 1 } = P$ and $P _ { 2 } = 0$ 2 we get

$$
\begin{array} { r l } & { \displaystyle \| P \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { 8 } ; \mathcal { V } ) } \leq \alpha ^ { - 1 } \sqrt { \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \| P \circ \hat { \mathcal { E } } _ { \mathcal { X } } ( X _ { i } ) \| _ { \mathcal { V } } ^ { 2 } } } \\ & { \leq \alpha ^ { - 1 } \sqrt { \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \| Y _ { i } - P \circ \hat { \mathcal { E } } _ { \mathcal { X } } ( X _ { i } ) \| _ { \mathcal { V } } ^ { 2 } } + \alpha ^ { - 1 } \sqrt { \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \| Y _ { i } \| _ { \mathcal { V } } ^ { 2 } } } \\ & { \leq \alpha ^ { - 1 } \sqrt { \displaystyle \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \| Y _ { i } - P \circ \hat { \mathcal { E } } _ { \mathcal { X } } ( X _ { i } ) \| _ { \mathcal { V } } ^ { 2 } } + \alpha ^ { - 1 } \| F \| _ { L _ { \varrho } ^ { \infty } ( \mathcal { X } ; \mathcal { V } ) } + \frac { \alpha ^ { - 1 } } { \sqrt { m } } \| E \| _ { 2 ; \mathcal { V } } . } \end{array}
$$

Substituting this into the previous expression and recalling that P was arbitrary now gives the result. □

With this to hand, we now establish the following general out-of-distribution error bound for the DNO training problem.

Theorem SM1.13 (Out-of-distribution generalization bound for the DNO training problem). Let $1 \leq k \leq d _ { \mathcal { X } } , \tau > 0 , \sigma \geq 1 , 0 < \epsilon < 1$ and N be the DNN model class defined above, where δ satisfies $\delta \le ( 4 \sqrt { k } ) ^ { - 1 }$ . Suppose that Assumption 5.1 holds and Assumption 5.3 holds with $\omega \geq 1$ satisfying $\omega _ { \mathrm { m i n } } = \mathrm { m i n } \{ \omega _ { j } \} >$ 1. Let $F \in \mathcal { H } ( b , \varepsilon , \mathcal { X } , \mathcal { Y } )$ , where b satisfies (4.2) for some $0 \textless p \textless 1$ , and draw $X _ { 1 } , \dots , X _ { m } \sim _ { \mathrm { i . i . d . } } \nu$ , where m satisfies

$$
m \geq c \cdot k \cdot \left( \log ^ { 4 } ( 2 k ) + \log ( 1 / \epsilon ) \right)
$$

for some universal constant $c > 0$ . Then, with probability at least $1 - \epsilon , a n y \ : ( \sigma , \tau ) \cdot$ approximate minimizer N<sup>ˆ</sup> of (5.3) is such that the approximation $\hat { F } = \hat { \mathcal { D } } _ { y } \circ \hat { N } \circ \hat { \mathcal { E } } _ { \mathcal { X } }$ satisfies

$$
\begin{array}{c} \| F - \hat { F } \| _ { L _ { \mu } ^ { \infty } ( \mathcal { X } ; \mathcal { Y } ) } \lesssim \frac { \sigma C ( b , \varepsilon , \omega , p ) } { \sqrt { \epsilon } } k ^ { 1 - 1 / p } + \frac { \sigma \sqrt { k } } { \sqrt { \epsilon } } \| F - \hat { B } _ { \mathcal { Y } } \circ F \| _ { L _ { \nu } ^ { 2 } ( \mathcal { X } ; \mathcal { Y } ) } + \frac { \sigma \sqrt { k } \| E \| _ { 2 ; \mathcal { Y } } } { \sqrt { m } }  \\ { + \sqrt { k } \tau + \sigma \delta k , } \end{array}
$$

where $E = ( E _ { i } ) _ { i = 1 } ^ { m } \in \mathcal { V } ^ { m }$ . Further, the class N satisfies

$$
\mathrm { w i d t h } ( \mathcal { N } ) \lesssim \frac { k \log ( k ) } { \log ( \omega _ { \mathrm { m i n } } ) } , \qquad \mathrm { d e p t h } ( \mathcal { N } ) \lesssim \log \left( \frac { \log ( k ) } { \log ( \omega _ { \mathrm { m i n } } ) } \right) .
$$

Proof. The width and depth bounds for $\mathcal { N }$ follow immediately from the construction and (SM1.37)-(SM1.38). We divide the remainder of the proof into a series of steps.

Step 1: Reduction to polynomials. Let $\hat { P }$ be as in Lemma SM1.12. Then (SM1.40)

$$
\begin{array} { r } { \| F - \hat { F } \| _ { L _ { \mu } ^ { \infty } ( \mathcal { X } ; \mathcal { Y } ) } \leq \| F - \hat { P } \circ \hat { \mathcal { E } } _ { \mathcal { X } } \| _ { L _ { \mu } ^ { \infty } ( \mathcal { X } ; \mathcal { Y } ) } + \| \hat { P } \circ \hat { \mathcal { E } } _ { \mathcal { X } } - \hat { D } _ { \mathcal { Y } } \circ \hat { N } \circ \hat { \mathcal { E } } _ { \mathcal { X } } \| _ { L _ { \mu } ^ { \infty } ( \mathcal { X } ; \mathcal { Y } ) } . } \end{array}
$$

For the second term, we recall first that $\hat { P }$ and $\hat { N }$ depend on the first $\lceil k \rceil \leq d _ { \mathcal { X } }$ variables only. Therefore, we can write

$$
\| \hat { P } \circ \hat { \mathcal { E } } _ { \mathcal { X } } - \hat { \mathcal { D } } _ { \mathcal { Y } } \circ \hat { N } \circ \hat { \mathcal { E } } _ { \mathcal { X } } \| _ { L _ { \mu } ^ { \infty } ( \mathcal { X } ; \mathcal { Y } ) } = \| \hat { P } - \hat { \mathcal { D } } _ { \mathcal { Y } } \circ \hat { N } \| _ { L _ { \mathcal { E } _ { \mathcal { X } } \sharp \mu } ^ { \infty } ( \mathbb { R } ^ { \mathbb { N } } ; \mathcal { Y } ) }
$$

We now apply Lemma SM1.11 to get

(SM1.41)

$$
\| \hat { P } \circ \hat { \mathcal { E } } _ { \mathcal { X } } - \hat { D } _ { \mathcal { Y } } \circ \hat { N } \circ \hat { \mathcal { E } } _ { \mathcal { X } } \| _ { L _ { \mu } ^ { \infty } ( \mathcal { X } ; \mathcal { Y } ) } \leq \delta \operatorname* { m a x } _ { S \in \mathcal { S } } \sqrt { | S | } \| P \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathbb { R } } ; \mathcal { Y } ) } \leq \delta \sqrt { k } \| \hat { P } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { n } ; \mathcal { Y } ) } .
$$

For the first term of (SM1.40), we write $f = F \circ \mathcal { D } _ { \mathcal { X } } : \mathbb { R } ^ { \infty }  \mathcal { Y }$ and recall that $\hat { P } \circ \hat { \mathcal { E } } _ { \mathcal { X } } = \hat { P } \circ \mathcal { E } _ { \mathcal { X } }$ to get

$$
\| F - \hat { P } \circ \hat { \mathcal { E } } _ { \mathcal { X } } \| _ { L _ { \mu } ^ { \infty } ( \mathcal { X } ; \mathcal { V } ) } = \| f - \hat { P } \| _ { L _ { \mathcal { E } _ { \mathcal { X } } \sharp \mu } ^ { \infty } ( \mathbb { R } ^ { \mathbb { N } } ; \mathcal { V } ) } .
$$

Hence

$$
\begin{array} { r } { \| \boldsymbol { F } - \boldsymbol { \hat { F } } \| _ { L _ { \mu } ^ { \infty } ( \mathcal { X } ; \mathcal { V } ) } \leq \| \boldsymbol { f } - \boldsymbol { \hat { P } } \| _ { L _ { \varepsilon _ { \mathcal { X } } \sharp \mu } ^ { \infty } ( \mathbb { R } ^ { \mathsf { N } } ; \mathcal { V } ) } + \delta \sqrt { k } \| \boldsymbol { \hat { P } } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathsf { N } } ; \hat { \mathcal { V } } ) } . } \end{array}
$$

Step $2 \colon { \hat { P } }$ is an approximate minimizer. Since $\hat { P }$ comes from Lemma SM1.12, it is a $( \sigma ^ { \prime } , \tau ^ { \prime } )$ -minimizer of (SM1.36), where $( \sigma ^ { \prime } , \tau ^ { \prime } )$ are given by (SM1.39). Recall that any $P \in \mathcal { P } _ { s ; \hat { \mathcal { y } } }$ depends on the first $\lceil k \rceil$ entries of an input ${ \pmb x } .$ . Since $k \leq d _ { \mathcal { X } }$ , we see that $P \circ \hat { \mathcal { E } } _ { \mathcal { X } } ( X _ { i } ) = P ( \pmb { x } _ { i } )$ , where $\pmb { x } _ { i } = \mathcal { E } _ { \pmb { \chi } } ( X _ { i } )$ is a uniformly distributed random variable on $D = ( - 1 , 1 ) ^ { \mathbb { N } }$ . We also have $F ( X _ { i } ) = f ( \pmb { x } _ { i } )$ and therefore $Y _ { i } = F ( X _ { i } ) + E _ { i }$ is equal to $y _ { i } = f ( \pmb { x } _ { i } ) + e _ { i }$ with $E _ { i } = e _ { i }$ . Hence (SM1.36) is equivalent to (SM1.24).

The conditions of Theorem SM1.7 hold by assumption, hence we may apply it with ϵ to deduce that

$$
\begin{array} { r l } & { \displaystyle \| f - \hat { P } \| _ { L _ { \mu } ^ { \infty } ( { \mathbb R } ^ { n } ; \mathcal { V } ) } \lesssim \frac { \sigma ^ { \prime } C ( b , \varepsilon , \omega , p ) } { \sqrt { \epsilon } } k ^ { 1 - 1 / p } + \frac { \sigma ^ { \prime } \sqrt { k } } { \sqrt { \epsilon } } \| f - \hat { B } _ { \mathcal { V } } \circ f \| _ { L _ { \varrho } ^ { 2 } ( { \mathbb R } ^ { n } ; \mathcal { V } ) } } \\ & { \quad \quad \quad + \frac { \sigma ^ { \prime } \sqrt { k } \| E \| _ { 2 ; \mathcal { V } } } { \sqrt { m } } + \sqrt { k } \tau ^ { \prime } , } \end{array}
$$

where $\pmb { { \cal E } } = ( \mathbb { R } _ { i } ^ { \mathbb { N } } ) _ { i = 1 } ^ { m }$ , with probability at least $1 - \epsilon / 2$ . Plugging in the values (SM1.39) for $\sigma ^ { \prime } , \tau ^ { \prime }$ combining with the previous bound, we deduce that

$$
\begin{array} { r l } & { \| F - \hat { F } \| _ { L _ { \mu } ^ { \infty } ( \mathcal { X } ; \mathcal { Y } ) } \lesssim \frac { \sigma ( 1 + \delta \sqrt { k } / \alpha ) C ( b , \varepsilon , \omega , p ) } { \sqrt { \epsilon } } k ^ { 1 - 1 / p } } \\ & { \qquad + \frac { \sigma ( 1 + \delta \sqrt { k } / \alpha ) \sqrt { k } } { \sqrt { \epsilon } } \| f - \hat { B } _ { \mathcal { Y } } \circ f \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { Y } ) } + \frac { \sigma ( 1 + \delta \sqrt { k } / \alpha ) \sqrt { k } \| E \| _ { 2 ; \mathcal { Y } } } { \sqrt { m } } } \\ & { \qquad + \sqrt { k } \tau + \frac { \sigma \delta k } { \alpha } \left( \| F \| _ { L _ { \nu } ^ { \infty } ( \mathcal { X } ; \mathcal { Y } ) } + \frac { 1 } { \sqrt { m } } \| E \| _ { 2 ; \mathcal { Y } } \right) } \\ & { \qquad + \delta k \| \hat { P } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { Y } ) } + \delta \sqrt { k } \| \hat { P } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { Y } ) } , } \end{array}
$$

where α is given by (SM1.25). Now recall from the proof of Theorem SM1. $7 \alpha \geq 1 / 2$ as part of the probabilistic event that holds with probability at least $1 - \epsilon$ that guarantees its main error bound. Since $\delta \leq 1 / ( 4 \sqrt { k } )$ by assumption, we obtain

(SM1.42)

$$
\begin{array} { r l } & { \displaystyle \| F - \hat { F } \| _ { L _ { \mu } ^ { \infty } ( \mathcal { X } ; \mathcal { Y } ) } \lesssim \frac { \sigma C ( b , \varepsilon , \omega , p ) } { \sqrt { \epsilon } } k ^ { 1 - 1 / p } + \frac { \sigma \sqrt { k } } { \sqrt { \epsilon } } \| f - \hat { B } _ { \mathcal { Y } } \circ f \| _ { L _ { \sigma } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { Y } ) } } \\ & { \qquad + \frac { \sigma \sqrt { k } \| E \| _ { 2 ; \mathcal { Y } } } { \sqrt { m } } + \sqrt { k } \tau + \sigma \delta k \left( \| F \| _ { L _ { \nu } ^ { \infty } ( \mathcal { X } ; \mathcal { Y } ) } + \frac { 1 } { \sqrt { m } } \| E \| _ { 2 ; \mathcal { Y } } \right) } \\ & { \qquad + \delta k \| \hat { P } \| _ { L _ { \sigma } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { Y } ) } . } \end{array}
$$

Step 3: Estimating $\| \hat { P } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathbb { N } } ; \mathcal { V } ) }$ The next step is to estimate the final term. Using (SM1.25) and the fact that $0 , \hat { P } \in \mathcal { P } _ { S ; \hat { \mathcal { V } } }$ , we get

$$
\begin{array} { r l r } {  { \| \hat { P } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } \leq \alpha ^ { - 1 } \sqrt { \frac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } { \| \hat { P } ( \pmb x _ { i } ) \| _ { \mathcal { V } } ^ { 2 } } } } } \\ & { } & { \leq \alpha ^ { - 1 } \sqrt { \frac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } { \| \hat { \mathcal D } _ { \mathcal { V } } \circ \hat { N } ( \pmb x _ { i } ) \| _ { \mathcal { V } } ^ { 2 } } } + \alpha ^ { - 1 } \| \hat { \mathcal D } _ { \mathcal { V } } \circ \hat { N } - \hat { P } \| _ { L _ { \varrho } ^ { \infty } ( \mathbb { R } ^ { n } ; \mathcal { V } ) } . } \end{array}
$$

We now use Lemma SM1.11 and the fact that $\hat { F } = \hat { \mathcal { D } } _ { y } \circ \hat { N } \circ \hat { \mathcal { E } } _ { \mathcal { X } }$ to obtain

$$
\| \hat { P } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathbb { N } } ; \mathcal { V } ) } \leq \alpha ^ { - 1 } \sqrt { \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \| \hat { F } ( \pmb { x } _ { i } ) \| _ { \mathcal { V } } ^ { 2 } } + \alpha ^ { - 1 } \delta \sqrt { k } \| \hat { P } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathbb { N } } ; \mathcal { V } ) } .
$$

We now rearrange and use the facts that $\hat { F }$ is as $( \sigma , \tau )$ -minimizer and the zero network belongs to $\mathcal { N } _ { ; }$ to get

$$
\begin{array} { r l } & { \| \hat { \boldsymbol { P } } \| _ { L _ { \varrho } ^ { 2 } ( { \mathbb { R } ^ { n } ; \mathcal { y } } ) } \leq \frac { \alpha ^ { - 1 } } { 1 - \alpha ^ { - 1 } \delta \sqrt { k } } \left( \sqrt { \frac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { m } \| \hat { \boldsymbol { F } } ( \boldsymbol { X } _ { i } ) - \boldsymbol { Y } _ { i } \| _ { \mathcal { y } } ^ { 2 } } + \frac { 1 } { \sqrt { m } } \| \boldsymbol { Y } \| _ { 2 ; \mathcal { y } } \right) } \\ & { \qquad \leq \frac { \alpha ^ { - 1 } } { ( 1 - \alpha ^ { - 1 } \delta \sqrt { k } ) } \left( \frac { 1 + \sigma } { \sqrt { m } } \| \boldsymbol { Y } \| _ { 2 ; \mathcal { y } } + \tau \right) . } \end{array}
$$

We now recall that $\alpha \ge 1 / 2 , \delta \le 1 / ( 4 \sqrt { k } )$ and $Y _ { i } = F ( X _ { i } ) + E _ { i }$ . This gives

$$
\| \hat { P } \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { N } ; \mathcal { V } ) } \lesssim ( 1 + \sigma ) \| F \| _ { L _ { \nu } ^ { \infty } ( \mathcal { X } ; \mathcal { V } ) } + \frac { 1 + \sigma } { \sqrt { m } } \| \pmb { E } \| _ { 2 ; \mathcal { V } } + \tau .
$$

Substituting this into (SM1.42) and recalling that $\delta \leq 1 / ( 4 \sqrt { k } )$ and $\sigma \geq 1$ we deduce that

$$
\begin{array} { l } { \displaystyle \| F - \hat { F } \| _ { L _ { \mu } ^ { \infty } ( \mathcal { X } ; \mathcal { Y } ) } \lesssim \frac { \sigma C ( b , \varepsilon , \omega , p ) } { \sqrt { \epsilon } } k ^ { 1 - 1 / p } + \frac { \sigma \sqrt { k } } { \sqrt { \epsilon } } \| f - \hat { B } _ { \mathcal { Y } } \circ f \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { n } ; \mathcal { Y } ) } + \frac { \sigma \sqrt { k } \| E \| _ { 2 ; \mathcal { Y } } } { \sqrt { m } } } \\ { \displaystyle \qquad + \sqrt { k } \tau + \sigma \delta k \| F \| _ { L _ { \nu } ^ { \infty } ( \mathcal { X } ; \mathcal { Y } ) } . } \end{array}
$$

After rewriting

$$
\| f - \hat { B } _ { \mathcal { V } } \circ f \| _ { L _ { \varrho } ^ { 2 } ( \mathbb { R } ^ { \mathbb { N } } ; \mathcal { V } ) } = \| F - \hat { B } _ { \mathcal { V } } \circ F \| _ { L _ { \nu } ^ { 2 } ( \mathcal { X } ; \mathcal { V } ) }
$$

and recalling that $\| F \| _ { L _ { \nu } ^ { \infty } ( \mathcal { X } ; \mathcal { V } ) } \le 1$ since $F \in \mathcal H ( b , \varepsilon ; \mathcal { X } , \mathcal { Y } )$ , we obtain the desired bound. □

## SM1.5.3. Final arguments.

Proof of Theorem 5.6. Let c be the constant from Theorem SM1.13 and suppose, without loss of generality, that $c \geq 1 / \log ^ { 4 } ( 2 )$ . We argue as in the proof of Theorem 4.1 and set k as in (SM1.35) and let $\bar { m } = \bar { m } ( \epsilon )$ be the smallest $m \in \mathbb { N } , m \geq 2$ , such that $k \geq 1$ . We now set $\delta = \operatorname* { m i n } \{ ( 4 \sqrt { k } ) ^ { - 1 } , 2 ^ { - m } / k \}$ and apply Theorem SM1.13 with $\sigma = 1$ and $\tau = 0$ . This completes the proof. □

Proof of Theorem $4 . 4$ . We will apply Theorem 5.6. Let $\mathcal { V } = \mathbb { R }$ with the Euclidean inner product and $d _ { \mathcal { V } } = 1$ , so that $\hat { \mathcal { D } } _ { y } = \hat { \mathcal { B } } _ { y } = \mathcal { I } _ { y }$ is the identity map. Let X be any separable, infinite-dimensional Hilbert space, $( \lambda _ { i } ) _ { i \in \mathbb { N } }$ be any summable sequence with $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \dots > 0$ and $\{ \phi _ { i } \} _ { i \in \mathbb { N } } \subset \mathcal { X }$ be any orthonormal basis. Define the probability measure ν on $\mathcal { X }$ as the pushforward $\nu = \mathcal { D } _ { \mathcal { X } } \sharp \varrho .$ Define the encoder $\hat { \mathcal { E } } _ { X } : X \mapsto$ $( \langle X , \phi _ { i } \rangle _ { \mathcal { X } } / \sqrt { \lambda _ { i } } ) _ { i = 1 } ^ { d _ { \mathcal { X } } }$ . It is clear that Assumption 5.1 holds. Given $\omega > 1$ with $\omega _ { \mathrm { m i n } } > 1$ let $\mathcal { N }$ be the DNN model class with $\begin{array} { r } { n _ { 0 } = d _ { \mathcal { X } } = \lceil c m / L \rceil } \end{array}$ and $n _ { M + 1 } = d y = 1$ whose existence is established by Theorem 5.6. Now let $f \in { \mathcal { H } } ( b , \varepsilon )$ for some b satisfying (4.2) with $0 < p < 1$ and $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { m } \sim _ { \mathrm { i . i . d . } } \varrho ,$ where $\varrho$ is the uniform probability measure on $( - 1 , 1 ) ^ { \mathbb { N } }$ . Let $X _ { i } = \mathcal { D } _ { \mathcal { X } } ( \pmb { x } _ { i } )$ and notice that $X _ { 1 } , \dots , X _ { m } \sim _ { \mathrm { i . i . d . } } \nu$ . Further, let $F = f \circ { \mathcal { E } } _ { \mathcal { X } }$ and observe that $F \in \mathcal H ( b , \varepsilon ; \mathcal { X } , \mathcal { Y } )$ . Then we can write

$$
y _ { i } = f ( \pmb { x _ { i } } ) + e _ { i } = F ( X _ { i } ) + e _ { i } = : F ( X _ { i } ) + E _ { i } = : Y _ { i }
$$

and observe that any minimizer $\hat { N }$ of (4.1) is also a minimizer of (5.3). Now let $\mu$ be an arbitrary probability measure on $\dot { \mathbb R } ^ { \mathbb N }$ that is supported in $D _ { \omega }$ . Define the measure $v = \mathcal { D } _ { \mathcal { X } } \sharp \mu$ on X and notice that υ satisfies Assumption 5.3. Observe that $\hat { F } = \hat { D } y \circ \hat { N } \circ \hat { \mathcal { E } } _ { \mathcal { X } } = \hat { N } \circ \mathcal { E } _ { \mathcal { X } }$ and therefore

$$
\| f - \hat { N } \| _ { L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { N } ) } = \| f \circ \mathcal { E } _ { \mathcal { X } } - \hat { N } \circ \mathcal { E } _ { \mathcal { X } } \| _ { L _ { \mathcal { D } _ { \mathcal { X } } \sharp \mu } ^ { \infty } ( \mathcal { X } ; \mathbb { R } ) } = \| F - \hat { F } \| _ { L _ { v } ^ { \infty } ( \mathcal { X } ; \mathbb { R } ) } ,
$$

since $\hat { \mathcal { E } } _ { { \mathcal X } } \circ { \mathcal D } _ { { \mathcal X } }$ is the identity on $\mathbb { R } ^ { \mathbb { N } }$ . We now apply the error bound from Theorem 5.6 and notice that the second term vanishes since $\hat { B } _ { \mathcal { Y } } = \mathcal { T } _ { \mathcal { Y } }$ □

SM2. Further details on the numerical experiments. We now give full details on the numerical experiments. First, we remark that all the experiments in the paper compute a certain error versus number of training samples $m .$ . We use multiple trials and, following [SM6, §A.1.3], we display the geometric mean and one geometric standard deviation. The number of trials is described below.

SM2.1. Polynomial estimators. For the experiments in §6.1, we compute the relative $L _ { \mu } ^ { \infty } ( \mathbb { R } ^ { d } )$ -norm error. We do this on a grid of size $1 0 ^ { 4 }$ , drawn randomly and independently from $\mu .$ . We use a total of 30 trials.

As mentioned, we employ an adaptive Least Squares (ALS) estimator. This follows [SM26] (see also [SM27], which is based on techniques for the constructing adaptive sparse grid quadratures [SM19]. ALS iteratively computes a sequence of least-squares polynomial approximations ${ \hat { f } } ^ { ( 1 ) } , { \hat { f } } ^ { ( 2 ) } , . . .$ . and corresponding (nested)

multi-index sets $S ^ { ( 1 ) } \subseteq S ^ { ( 2 ) } \subseteq \cdots \subset \mathbb { N } _ { 0 } ^ { d }$ , where $\hat { f } ^ { ( l ) } \in \mathcal { P } _ { S ^ { ( l ) } } , \forall l \in \mathbb { N }$ and ${ \mathcal { P } } _ { S ^ { ( l ) } }$ is as in (2.3).

We now describe how the one-step update in ALS is performed. This requires several definitions. First, we define the margin of a multi-index set $S \subseteq \mathbb { N } _ { 0 } ^ { d }$ as

$$
\begin{array} { r } { \mathcal { M } ( S ) = \left\{ \pmb { n } \in \mathbb { N } _ { 0 } ^ { d } \backslash S : \exists j \in [ d ] : \pmb { n } - \pmb { e } _ { j } \in S \right\} , } \end{array}
$$

where $e _ { j }$ is jth canonical multi-index, and the reduced margin of S as the set

$$
\mathscr { R } ( S ) = \{ \pmb { n } = ( \nu _ { j } ) _ { j = 1 } ^ { d } \in \mathscr { M } ( S ) : \forall j \in [ d ] , \nu _ { j } \neq 0 \ \Rightarrow \ \pmb { n } - \pmb { e } _ { j } \in S \} .
$$

Given a current multi-index set $S \subset \mathbb { N } _ { 0 } ^ { d }$ , ALS constructs a new set by selecting multiindices $\mathcal { R } ( S )$ using a bulk chasing procedure, defined as follows. Let $e : \mathcal { R } ( S ) $ R and $0 < \beta \le 1$ be a parameter. The function e serves as an estimate for the true polynomial coeficients of the function being approximated. We describe a precise choice of e below. With this, the procedure bulk $( \mathcal { R } ( S ) , e , \beta )$ computes a set $T \subseteq { \mathcal { R } } ( S )$ of minimal positive cardinality such that

$$
\sum _ { \pmb { n } \in T } e ( \pmb { n } ) \geq \beta \sum _ { \pmb { n } \in \mathcal { R } ( S ) } e ( \pmb { n } ) .
$$

Now suppose at some iteration we have an multi-index set $S \subset { \mathcal { F } }$ and corresponding least-squares approximation $\hat { f } \in \mathcal { P } _ { S }$ based on sample points $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { m }$ . We define $e : \mathcal { R } ( S )  \mathbb { R }$ as the discrete residual

$$
e ( \pmb { n } ) = \left| \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left( f ( \pmb { x } _ { i } ) - \hat { f } ( \pmb { x } _ { i } ) \right) \Psi _ { \pmb { n } } ( \pmb { x } _ { i } ) \right| ^ { 2 } , \quad \forall \pmb { n } \in \mathcal { R } ( S ) .\tag{SM2.1}
$$

We now compute $T = \mathrm { b u l k } ( \mathcal { R } ( S ) , e , \beta )$ , construct the new index set $S _ { \mathsf { n e w } } = S \cup T$ and compute a new estimator $\hat { f } _ { \sf n e w }$ as the least-squares polynomial approximation to f from $\mathcal { P } _ { S _ { \mathrm { n e w } } }$

The full ALS approximation is presented in Algorithm SM2.1. Note that, following [SM27], we use the value $\beta = 0 . 5$ in our experiments. We also make the choice $\begin{array} { r } { \begin{array} { r } { \min ^ { ( l ) } = c _ { d } \operatorname* { m a x } \{ n ^ { ( l ) } + 1 , \lceil n ^ { ( l ) } \cdot \log ( n ^ { ( l ) } ) \rceil \} } \end{array} } \end{array}$ , where $c _ { d } = 2$ when $d = 8$ and $c _ { d } = 1$ otherwise. The reason for the larger constant in lower dimensions is to avoid instability of the least-squares system.

SM2.2. DNN estimators. For the experiments in $\ S \ 6 . 2$ , we compute the relative $L _ { \mu } ^ { 2 } ( \mathbb { R } ^ { d } )$ -norm error. We do this on a grid of size $1 0 ^ { 4 }$ , drawn randomly and independently from $\mu .$ . We use fully connected DNNs with the architecture and parameters specified in Table SM1. For each training-set size, we perform 30 independent trials.

SM2.3. DNO estimators. We now describe the DNO setup for the experiments in §6.3. As noted we use FNOs. Our setup and implementation is based on [SM22, SM23].

Darcy flow. Details of the architecture and training parameters are given in Table SM2. In our experiments, we show the discrete relative $L _ { \mu } ^ { 2 }$ -norm error, computed over the computational grid. For a fixed input-output pair $( \dot { a } , u )$ , this is defined as

$$
e _ { \mathrm { r e l } } = \frac { \left( \sum _ { p , q } | \widetilde { u } _ { p , q } - u _ { p , q } | ^ { 2 } \right) ^ { 1 / 2 } } { \left( \sum _ { p , q } | u _ { p , q } | ^ { 2 } \right) ^ { 1 / 2 } } ,
$$

Algorithm SM2.1 Adaptive least-squares approximation   
Require: Function to approximate $f \in L _ { \varrho } ^ { 2 } ( [ - 1 , 1 ] ^ { d } )$ , bulk chasing parameter $0 ~ <$   
$\beta \leq 1$   
Ensure: Sequence of multi-index sets $S ^ { ( 1 ) } \subseteq S ^ { ( 2 ) } \subseteq \cdots \subset \mathbb { N } _ { 0 } ^ { d }$ and approximations   
${ \hat { f } } ^ { ( 1 ) } , { \hat { f } } ^ { ( 2 ) }$   
Set ${ \dot { S } } ^ { ( 1 ) } = \{ \mathbf { 0 } \}$   
for $l = 1 , 2 , \ldots$ do   
Set $n ^ { ( l ) } = \vert S ^ { ( l ) } \vert$   
Determine a number of samples $m ^ { ( l ) } \geq n ^ { ( l ) }$   
Draw $\pmb { x } _ { 1 } , \dots , \pmb { x } _ { m ^ { ( l ) } } \sim _ { \mathrm { i . i . d } }$ <sub>.</sub> ϱ   
Compute $\begin{array} { r } { \hat { f } ^ { ( l ) } \in \underset { - \infty ^ { ( l ) } } { \mathrm { a r g m i n } } \frac { 1 } { m ^ { ( l ) } } \sum _ { i = 1 } ^ { m ^ { ( l ) } } | f ( \pmb { x } _ { i } ) - p ( \pmb { x } _ { i } ) | ^ { 2 } } \end{array}$   
$ { \boldsymbol { p } } \in  { \mathcal { P } } ^ { ( l ) }$   
Compute the estimator $e = e ^ { ( l ) }$ via (SM2.1) with $m = m ^ { ( l ) } , S = S ^ { ( l ) }$ and ${ \hat { f } } = { \hat { f } } ^ { ( l ) }$   
Compute $T ^ { ( l ) } = \tt b u l k ( \mathcal { R } ( S ^ { ( l ) } ) ,  e ^ { ( l ) } , \beta )$   
Set ${ \hat { S } } ^ { ( l + 1 ) } = S ^ { ( l ) } \cup T ^ { ( l ) }$   
end for   
Parameter Value   
Hidden layers 15   
Hidden layer width 150   
Activation tanh   
Training samples {10, 20, 40, 80, 160, 320, 640, 1280}   
Test samples 10,000   
Batch size Full batch (all training samples)   
Epochs 2000   
Optimizer Adam   
Weight decay 0   
Initial learning rate $1 0 ^ { - 4 }$   
Learning-rate schedule Multiplied by 0.999 every epoch  
Table SM1: DNN architecture and training parameters for benchmark functions from the Virtual Library of Simulation Experiments.

where $\widetilde { u }$ denotes the FNO prediction, and $u _ { p , q }$ and $\widetilde { u } _ { p , q }$ are the grid values of $u$ and $\tilde { u } ,$ respectively.

The test distribution $\mu ^ { \mathrm { c o o k i e s } }$ is based on a benchmark commonly known as the fixed-radius cookie problem [SM10, SM9, SM12]. Let $D \ = \ [ 0 , 1 ] ^ { 2 }$ , and let $\Omega _ { i } .$ $i = 1 , \ldots , 8 .$ , be non-overlapping disks of radius $r _ { \mathrm { c o o k i e } } ,$ with centres $( 0 . 5 \pm 0 . 3 , 0 . 5 \pm$ $( 0 . 3 ) , ( 0 . 5 , 0 . 5 \pm 0 . 3 ) , ( 0 . 5 \pm 0 . 3 , 0 . 5 )$ . Then $a \sim \mu ^ { \mathrm { c o o k i e s } }$ if

$$
a ( x ) = 1 - \sum _ { i = 1 } ^ { 8 } \mathbf { 1 } _ { \Omega _ { i } } ( x ) \left( C _ { 1 } + C _ { 2 } y _ { i } \right) , \quad { \mathrm { w h e r e ~ } } y _ { 1 } , \dots , y _ { 8 } \sim _ { \mathrm { i } \cdot \mathrm { i } \cdot \mathrm { d } } . \mathcal { U } ( [ - 1 , 1 ] ) .\tag{SM2.2}
$$

In our experiments, we set $r _ { \mathrm { c o o k i e } } = 0 . 1 4 , C _ { 1 } = 0 . 6 2 5 , C _ { 2 } = 0 . 3 7 5 .$

Fig. SM1 illustrates selected input–output pairs drawn from the diferent training and test distributions used in our experiments.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Training resolution</td><td> $6 4 \times 6 4$ </td></tr><tr><td>Testing resolution</td><td> $6 4 \times 6 4$ </td></tr><tr><td>Training samples</td><td>{10, 20, 40, 80, 160, 320, 640, 1280}</td></tr><tr><td>Test samples</td><td>200</td></tr><tr><td>Input channels</td><td>1</td></tr><tr><td>Output channels</td><td>1</td></tr><tr><td>Fourier layers</td><td>4</td></tr><tr><td>Fourier modes</td><td>12</td></tr><tr><td>Hidden channels</td><td>32</td></tr><tr><td>Projection channels</td><td>2</td></tr><tr><td>Activation</td><td>GeLU</td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Epochs</td><td>100</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Weight decay</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Initial learning rate</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Learning-rate schedule</td><td>Halved every 100 epochs</td></tr></table>

Table SM2: FNO architecture and training parameters for the Darcy Flow problem.

![](images/3bf4eda8e3de202f454eccd085f8cfe9939b2d62d242bbf4c7e89497e1056624.jpg)  
Fig. SM1: Representative input-output pairs from the Darcy flow training and test distributions. The top row shows the difusion coeficients $^ { a , }$ and the bottom row shows the corresponding solutions u. The distributions are $\mu _ { 3 , 2 } ^ { \mathrm { L N } } , \mu _ { 3 , 1 } ^ { \mathrm { L N } } , \mu ^ { \mathrm { c o o k i e s } }$ and $\mu _ { 3 , 1 } ^ { \mathrm { P C } }$ (left to right).

Incompressible Navier–Stokes. Details of the architecutre and training parameters are given in Table SM3. In our experiments, we show the discrete relative $L _ { \mu } ^ { 2 } .$ -norm error, computed jointly spatio-temporal computational grid. For a fixed input-output pair (w<sub>0</sub>, w), this is defined as

$$
e _ { \mathrm { r e l } } = \frac { \left( \sum _ { p , q , l } { \left| \widetilde { w } _ { p , q , l } - w _ { p , q , l } \right| ^ { 2 } } \right) ^ { 1 / 2 } } { \left( \sum _ { p , q , l } { \left| w _ { p , q , l } \right| ^ { 2 } } \right) ^ { 1 / 2 } } ,
$$

where $\widetilde { w }$ denotes the FNO prediction, and $w _ { p , q , l }$ and $\widetilde { w } _ { p , q , l }$ are the grid values of w and $\widetilde { w } .$ , respectively.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Training resolution</td><td> $6 4 \times 6 4 \times 5 0$ </td></tr><tr><td>Testing resolution</td><td> $6 4 \times 6 4 \times 5 0$ </td></tr><tr><td>Training samples</td><td>[150, 300, 600, 1200, 2400, 4800]</td></tr><tr><td>Test samples</td><td>200</td></tr><tr><td>Input channels</td><td>4</td></tr><tr><td>Output channels</td><td>1</td></tr><tr><td>Fourier layers</td><td>4</td></tr><tr><td>Fourier modes</td><td>12</td></tr><tr><td>Hidden channels</td><td>32</td></tr><tr><td>Projection channels</td><td>2</td></tr><tr><td>Activation</td><td>GeLU</td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Epochs</td><td>200</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Weight decay</td><td>0</td></tr><tr><td>Initial learning rate</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Learning-rate schedule</td><td>Halved every 50 epochs</td></tr></table>

Table SM3: FNO architecture and training parameters for the Navier-Stokes problem.

## REFERENCES

[SM1] M. Abramowitz and I. A. Stegun, eds., Handbook of Mathematical Functions: With Formulas, Graphs, and Mathematical Tables, Dover Publications, Inc., New York, NY, 1964.

[SM2] B. Adcock, S. Brugiapaglia, N. Dexter, and S. Moraga, Deep neural networks are efective at learning high-dimensional Hilbert-valued functions from limited data, in Proceedings of The Second Annual Conference on Mathematical and Scientific Machine Learning, J. Bruna, J. S. Hesthaven, and L. Zdeborov´a, eds., vol. 145 of Proc. Mach. Learn. Res. (PMLR), PMLR, 2021, pp. 1–36.

[SM3] B. Adcock, S. Brugiapaglia, N. Dexter, and S. Moraga, Learning smooth functions in high dimensions: from sparse polynomials to deep neural networks, in Numerical Analysis Meets Machine Learning, S. Mishra and A. Townsend, eds., vol. 25 of Handbook of Numerical Analysis, Elsevier, 2024, pp. 1–52.

[SM4] B. Adcock, S. Brugiapaglia, N. Dexter, and S. Moraga, On eficient algorithms for computing near-best polynomial approximations to high-dimensional, Hilbert-valued functions from limited samples, vol. 13 of Mem. Eur. Math. Soc., EMS Press, 2024.

[SM5] B. Adcock, S. Brugiapaglia, N. Dexter, and S. Moraga, Near-optimal learning of Banach-valued, high-dimensional functions via deep neural networks, Neural Networks, 181 (2025), p. 106761.

[SM6] B. Adcock, S. Brugiapaglia, and C. G. Webster, Sparse Polynomial Approximation of High-Dimensional Functions, Comput. Sci. Eng., Society for Industrial and Applied Mathematics, Philadelphia, PA, 2022.

[SM7] B. Adcock, N. Dexter, and S. Moraga, Optimal deep learning of holomorphic operators between Banach spaces, in Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, eds., vol. 37, Curran Associates, Inc., 2024, pp. 27725–27789.

[SM8] B. Adcock, N. Dexter, and S. Moraga, Optimal approximation of infinite-dimensional holomorphic functions II: recovery from iid pointwise samples, J. Complexity, 89 (2025), p. 101933.

[SM9] J. Back, F. Nobile, L. Tamellini, and R. Tempone<sup>¨</sup> , Stochastic spectral Galerkin and collocation methods for PDEs with random coeficients: a numerical comparison, in Spectral and High Order Methods for Partial Diferential Equations, J. S. Hesthaven and E. M. Rønquist, eds., vol. 76 of Lect. Notes Comput. Sci. Eng., Berlin, Heidelberg, Germany, 2011, Springer, pp. 43–62.

[SM10] J. Ballani and L. Grasedyck, Hierarchical tensor approximation of output quantities of parameter-dependent pdes, SIAM/ASA Journal on Uncertainty Quantification, 3 (2015), pp. 852–872, https://doi.org/10.1137/140960980.

[SM11] S. Brugiapaglia, S. Dirksen, H. C. Jung, and H. Rauhut, Sparse recovery in bounded Riesz systems with applications to numerical methods for PDEs, Appl. Comput. Harmon. Anal., 53 (2021), pp. 231–269.

[SM12] A. Chkifa, A. Cohen, G. Migliorati, F. Nobile, and R. Tempone, Discrete least squares polynomial approximation with random evaluations - application to parametric and stochastic elliptic PDEs., ESAIM Math. Model. Numer. Anal., 49 (2015), pp. 815–837.

[SM13] A. Chkifa, A. Cohen, and C. Schwab, Breaking the curse of dimensionality in sparse polynomial approximation of parametric PDEs, J. Math. Pures Appl., 103 (2015), pp. 400– 428.

[SM14] A. Chkifa, N. Dexter, H. Tran, and C. G. Webster, Polynomial approximation via compressed sensing of high-dimensional functions on lower sets, Math. Comp., 87 (2018), pp. 1415–1450.

[SM15] A. Cohen and R. A. DeVore, Approximation of high-dimensional parametric PDEs, Acta Numer., 24 (2015), pp. 1–159.

[SM16] A. Cohen, R. A. DeVore, and C. Schwab, Convergence rates of best N-term Galerkin approximations for a class of elliptic sPDEs, Found. Comput. Math., 10 (2010), pp. 615– 646.

[SM17] A. Cohen, R. A. DeVore, and C. Schwab, Analytic regularity and polynomial approximation of parametric and stochastic elliptic PDE’s, Anal. Appl. (Singap.), 9 (2011), pp. 11–47.

[SM18] T. De Ryck, S. Lanthaler, and S. Mishra, On the approximation of functions by tanh neural networks, Neural Networks, 143 (2021), pp. 732–750.

[SM19] T. Gerstner and M. Griebel, Dimension-adaptive tensor-product quadrature, Computing, 71 (2003), pp. 65–87.

[SM20] I. Guhring and M. Raslan <sup>¨</sup> , Approximation rates for neural networks with encodable weights in smoothness spaces, Neural Networks, 134 (2021), pp. 107–130.

[SM21] B. Li, S. Tang, and H. Yu, Better approximations of high dimensional smooth functions by deep neural networks with rectified power units, Commun. Comput. Phys., 27 (2020), pp. 379–411.

[SM22] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar, Fourier neural operator for parametric partial diferential equations, in ICLR, 2021.

[SM23] Z. Li, H. Zheng, N. B. Kovachki, D. Jin, H. Chen, B. Liu, K. Azizzadenesheli, and A. Anandkumar, Physics-informed neural operator for learning partial diferential equations, ACM/IMS J. Data Sci., 1 (2024), pp. 1–27.

[SM24] J. Lu, Z. Shen, H. Yang, and S. Zhang, Deep network approximation for smooth functions, SIAM J. Math. Anal., 53 (2021), pp. 5465–5506.

[SM25] H. Mhaskar, Neural networks for optimal approximation of smooth and analytic functions., Neural Comput., 8 (1996), pp. 164–177.

[SM26] G. Migliorati, Adaptive polynomial approximation by means of random discrete least squares, in Numerical Mathematics and Advanced Applications – ENUMATH 2013, A. Abdulle, S. Deparis, D. Kressner, F. Nobile, and M. Picasso, eds., Cham, Switzerland, 2015, Springer, pp. 547–554.

[SM27] G. Migliorati, Adaptive approximation by optimal weighted least squares methods, SIAM J. Numer. Anal, 57 (2019), pp. 2217–2245.

[SM28] J. A. A. Opschoor, C. Schwab, and J. Zech, Exponential ReLU DNN expression of holomorphic maps in high dimension, Constr. Approx., 55 (2022), pp. 537–582.

[SM29] H. Rauhut and C. Schwab, Compressive sensing Petrov-Galerkin approximation of highdimensional parametric operator equations, Math. Comp., 86 (2017), pp. 661–700.

[SM30] H. Rauhut and R. Ward, Interpolation via weighted ℓ<sup>1</sup> minimization, Appl. Comput. Harmon. Anal., 40 (2016), pp. 321–351.

[SM31] C. Schwab and J. Zech, Deep learning in high dimension: neural network expression rates for generalized polynomial chaos expansions in UQ, Anal. Appl. (Singap.), 17 (2019), pp. 19–55.

[SM32] C. Schwab and J. Zech, Deep learning in high dimension: neural network expression rates for analytic functions in $L ^ { 2 } ( \mathbb { R } ^ { d } , \gamma _ { d } )$ , SIAM/ASA J. Uncertain. Quantif., 11 (2023), pp. 199–234.

[SM33] D. Yarotsky, Error bounds for approximations with deep ReLU networks, Neural Networks, 94 (2017), pp. 103–114.