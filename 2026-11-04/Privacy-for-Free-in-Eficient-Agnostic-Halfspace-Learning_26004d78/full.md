# Privacy for Free in Eficient Agnostic Halfspace Learning

Giorgio Racca, Amartya Sanyal, and Vikrant Singhal

Department of Computer Science, University of Copenhagen {g.racca,amsa,vikrant.singhal}@di.ku.dk

## Abstract

We provide the first computationally eficient, diferentially private algorithms for agnostically learning halfspaces. In particular, we study three diferent settings of agnostic learning: improper learning under Gaussian and log-concave marginals; proper learning under Gaussian marginals; and semi-agnostic learning under Gaussian marginals. In all three aforementioned settings, the added cost in the sample complexity due to privacy matches the respective non-private sample complexity, implying that we get privacy for free.

## 1 Introduction

Learning halfspaces is a classical problem in learning theory. A halfspace is specified by a weight vector $w \in \mathbb { R } ^ { d }$ and a threshold $t \in \mathbb { R }$ , and classifies a point $\boldsymbol { x } \in \mathbb { R } ^ { d }$ according to sign $( \left. w , x \right. + t )$ . We write P for the distribution of labeled examples and $\mu = P _ { X }$ for its input marginal. In classical realizable PAC learning, the distribution P is assumed to be linearly separable and the learner is required to output a classifier h with err $\mathbf { \Omega } _ { P } ( h ) \leq \alpha$ for any accuracy parameter $\alpha \in ( 0 , 1 )$ . In the agnostic setting, the labels in the dataset may be arbitrary. Thus, the guarantee is relaxed to requiring the learner to output a classifier h with error

$$
\mathrm { e r r } _ { P } ( h ) \leq \mathrm { O P T } + \alpha ,
$$

where OPT is the minimum error achievable by a halfspace under P. This setting allows the data to be noisy or imperfectly separable by a linear decision boundary. Diferential privacy (DP) (Dwork et al., 2006) imposes an additional constraint, requiring the distribution of the learner’s output to change only slightly when a single training example is replaced. In this paper, we study whether agnostic halfspace learning can be made diferentially private with small computational and statistica overhead relative to non-private learning.

Agnostically learning halfspaces is known to be computationally hard, even for non-private improper learners, i.e., learners whose output need not be a halfspace (Tiegel, 2023). To overcome this, a rich line of work has focused on distributions with structured marginals but arbitrary labels. Under Gaussian and more generally log-concave marginals, Kalai et al. (2008) provided the first algorithm for agnostic learning of halfspaces whose runtime is polynomial in the dimension for every fixed accuracy. For Gaussian marginals, their complexity scales as $d ^ { \mathcal { O } \left( 1 / \alpha ^ { 4 } \right) }$ which was later improved to $d ^ { \mathcal { O } \left( 1 / \alpha ^ { 2 } \right) }$ by Diakonikolas et al. (2010). However, their algorithm was improper. A proper algorithm must output a halfspace and can be desirable because a linear halfspace ofers a more compact and interpretable representation of the classifier with $\mathcal O \left( d \right)$ parameters. For Gaussian marginals, Diakonikolas et al. (2021a) provided the first agnostic proper learner for halfspaces with somewhat comparable statistical and computational complexity. For DP agnostic learning, Kasiviswanathan et al. (2008) provided the first agnostic private learners for finite classes with sample complexity scaling logarithmically in the size of the hypothesis class. Alon et al. (2020) and Li et al. (2025) provided general reductions from agnostic learning to realisable learning. However, their reductions are not shown to be computationally eficient. For halfspaces over finite domains, Beimel et al. (2019) provided eficient private agnostic learners, but over infinite domains such as $\mathbb { R } ^ { d }$ , distribution-free private learning of halfspaces is information-theoretically impossible even in the realisable setting due to their infinite Littlestone dimension (Bun et al., 2015; Alon et al., 2019). Distributional assumptions on the marginal $\mu$ are therefore essential for both privacy and computational eficiency. Given the non-private agnostic learners under structured marginals (Kalai et al., 2008; Diakonikolas et al., 2021a), our goal in this work is to obtain matching statistical and computational complexities under diferential privacy.

<table><tr><td>Marginal</td><td>Learner</td><td>Error</td><td>Sample complexity  $\widetilde { \mathcal { O } } \left( \cdot \right)$ </td></tr><tr><td>Gaussian</td><td>Improper</td><td> $\mathrm { O P T } + \alpha$ </td><td> $\frac { D _ { \mathrm { G } } } { \alpha ^ { 2 } } + \frac { D _ { \mathrm { G } } } { \varepsilon _ { \mathrm { D P } } \alpha }$ </td></tr><tr><td>Log-concave</td><td>Improper</td><td> $\mathrm { O P T } + \alpha$ </td><td> $\frac { D _ { \mathrm { { L } } } } { \alpha ^ { 2 } } + \frac { D _ { \mathrm { { L } } } } { \varepsilon _ { \mathrm { { D P } } } \alpha ^ { 3 / 2 } }$ </td></tr><tr><td>Gaussian</td><td>Proper</td><td> $\mathrm { O P T } + \alpha$ </td><td> $\frac { D _ { \mathrm { G } } } { \alpha ^ { 2 } } + \frac { D _ { \mathrm { G } } } { \varepsilon _ { \mathrm { D P } } \alpha } + \frac { 1 } { \alpha ^ { 6 } } + \frac { 1 } { \varepsilon _ { \mathrm { D P } } \alpha ^ { 5 } }$ </td></tr><tr><td>Gaussian</td><td>Proper</td><td> $C { \cdot } \mathrm { O P T } { + } \alpha$ </td><td> $\frac { d } { \alpha ^ { 2 } } + \frac { d } { \varepsilon _ { \mathrm { D P } } \alpha }$ </td></tr></table>

Table 1: The rows correspond, respectively, to Theorem 2.1, Theorem D.1, Theorem 3.1, and Theorem 4.1. Here, $D _ { \mathrm { G } } \asymp d ^ { \mathcal { O } \left( 1 / \alpha ^ { 2 } \right) }$ and $D _ { \mathrm { L } } \asymp d ^ { \mathrm { e x p } ( \mathcal { O } ( 1 / \alpha ) ) }$ match known rates in the non-private setting (Diakonikolas et al., 2010; Kalai et al., 2008), and $\widetilde { \mathcal { O } }$ suppresses polylogarithmic factors.

However, even under Gaussian marginals, the $d ^ { \mathcal { O } \left( 1 / \alpha ^ { 2 } \right) }$ dependence on statistical and therefore computational complexity is likely unavoidable for approaches using polynomial regression due to known SQ lower bounds (Diakonikolas et al., 2021b; Goel et al., 2020). A natural relaxation for the problem is semi-agnostic proper learning, which allows halfspaces h with

$$
\mathrm { e r r } _ { P } ( h ) \leq C \cdot \mathrm { O P T } + \alpha ,
$$

where C is a known universal constant. Non-private learners, under Gaussian marginals, can achieve this guarantee in time polynomial in both d and $1 / \alpha$ simultaneously (Diakonikolas et al., 2022). This raises the natural question of whether the polynomial dependence on $d , 1 / \alpha$ can be maintained for semi-agnostic learning under privacy. Together, these lead to the following question:

What statistical and computational guarantees can DP algorithms get under Gaussian marginals for i) agnostic improper, ii) agnostic proper, and $i i i )$ semi-agnostic proper learning of halfspaces ?

## Our Results

Table 1 summarizes our results. All algorithms satisfy $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ guarantees with respect to the dataset, and their error guarantees hold with high probability. Additionally, all our algorithms run in time polynomial in the sample size for every fixed accuracy. Throughout, $\widetilde { \mathcal { O } } \left( \cdot \right)$ hides factors polylogarithmic in $1 / \alpha , 1 / \beta , 1 / \varepsilon _ { \mathrm { D P } }$ , and $1 / \delta _ { \mathrm { D P } }$

Agnostic Improper learning. Our first result in Section 2 gives a private improper agnostic learner under the standard Gaussian marginal. The algorithm relies on private polynomial $L ^ { 1 }$ regression using clipped Hermite features of degree $\mathcal { O } \left( 1 / \alpha ^ { 2 } \right)$ . A key step is to control the efect of clipping on the error and then convert the expected excess-loss bound to a high-probability one. This achieves an error at most $\mathrm { O P T } + \alpha$ with high probability using sample complexity

$$
n = \widetilde { \mathcal { O } } \left( \frac { D _ { p } } { \alpha ^ { 2 } } + \frac { D _ { p } } { \alpha \varepsilon _ { \mathrm { D P } } } \right) \mathrm { w h e r e } D _ { p } = \binom { d + p } { p }\tag{1}
$$

The first term in Equation (1) is the statistical cost of the problem due to finite samples and the second term is the cost due to privacy. In Appendix D, we extend this approach to known log-concave marginals using $p = \exp \left( \mathcal { O } \left( 1 / \alpha \right) \right)$ , incurring an additional $\alpha ^ { - 1 / 2 }$ factor in the privacy term of Equation (1) (as shown in Table 1).

Agnostic Proper learning. In Section 3, we give a proper DP learner for halfspaces under Gaussian marginals with error at most $\mathrm { O P T } + \alpha$ . As shown in Table 1, its sample size is

$$
n = \widetilde { \mathcal { O } } \left( \frac { D _ { p } } { \alpha ^ { 2 } } + \frac { D _ { p } } { \alpha \varepsilon _ { \mathrm { D P } } } + \frac { 1 } { \alpha ^ { 6 } } + \frac { 1 } { \varepsilon _ { \mathrm { D P } } \alpha ^ { 5 } } \right)\tag{2}
$$

The learner works by first estimating a polynomial privately, similar to the improper case, which gives the first two terms in Equation (2). Then we show that this polynomial can be used to obtain a subspace of dimension $\mathcal { O } \left( \alpha ^ { - 4 } \right)$ which contains a halfspace with error close to $\mathrm { O P T }$ . Finally, creating a discrete cover of halfspaces supported on this space, and using the exponential mechanism leads to the final output. This subspace creation, discretisation, and exponential mechanism costs the last two terms in Equation (2).

Semi-Agnostic Proper learning. In Section 4, we give a proper learner under Gaussian marginals with error at most $\mathcal { O } \left( \mathrm { O P T } \right) + \alpha$ . The sample complexity, as described in Table 1, is

$$
n = \widetilde { \mathcal { O } } \left( \frac { d } { \alpha ^ { 2 } } + \frac { d } { \alpha \varepsilon _ { \mathrm { D P } } } \right) .\tag{3}
$$

The learner works by utilizing a privatized semi-agnostic learner from Diakonikolas et al. (2022) that provides the error guarantee of $\mathcal { O } \left( \eta \right)$ , where $\eta \geq \mathrm { O P T }$ is known a priori. However, since we do not know what $\eta$ is, we search over a small candidate set of $\eta \mathrm { { s } }$ by using the said privatized learner, and then pick the best hypothesis using the exponential mechanism on a fresh validation set.

Remark 1. Note that $d ^ { O ( 1 / \alpha ^ { 2 } ) } \cdot 1 / \alpha ^ { c } = d ^ { O ( 1 / \alpha ^ { 2 } ) }$ . We fix p for both improper and proper agnostic settings, such that it is the smallest $p \in \mathbb N$ guaranteeing the existence of the polynomial bases that accurately approximate halfspaces. In other words, we use the same p as required by the known non-private counterparts of our algorithms (Diakonikolas et al., 2010; Pesenti et al., 2026). Thus, whatever exponent d has in the sample complexity of the respective non-private learners, we achieve that, while keeping the additional cost in poly $( 1 / \alpha )$ explicitly outside the exponent of d.

Remark 2. We additionally note that the added cost in our sample complexity due to DP matches the non-private sample complexity (terms without $\varepsilon _ { \mathrm { D P } }$ or $\delta _ { \mathrm { D P } } )$ for each of the three Gaussian settings above when $\varepsilon _ { \mathrm { D P } } = \Omega ( \alpha )$ . In that sense, we get privacy for free. This is commonly observed in DP statistical estimation (see, e.g., Karwa and Vadhan (2018); Kamath et al. (2019)), and such sample complexity is often tight for various problems, such as mean and covariance estimation (Kamath et al., 2022).

Notation. We denote the class of halfspaces by C. For a classifier $h \colon  { \mathbb { R } } ^ { d } \to \{ \pm 1 \}$ , its population error is err $_ P ( h ) : = \mathbb { P } _ { ( X , Y ) \sim P } [ h ( X ) \neq Y ]$ and, given a sample S, its empirical error is ${ \widehat { \mathrm { e r r } } } _ { S } ( h ) : = $ $\textstyle | S | ^ { - 1 } \sum _ { ( x , y ) \in S } \mathbf { 1 } \{ h ( x ) \neq y \}$ . Unless stated otherwise, the input marginal is $\mu = \mathcal { N } ( 0 , I _ { d } )$ ; the only exception is our result for log-concave marginals (Theorem D.1). We define $\begin{array} { r } { \mathrm { O P T } : = \operatorname* { i n f } _ { c \in \mathcal { C } } \mathrm { e r r } _ { P } ( c ) } \end{array}$ For a function $g \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ and $q \in \{ 1 , 2 \}$ , we write $\| g \| _ { L ^ { q } ( \mu ) } : = ( \mathbb { E } _ { X \sim \mu } [ | g ( X ) | ^ { q } ] ) ^ { 1 / q }$ . The formal definition of diferential privacy and associated properties are given in Appendix B.2.

## 2 Private Improper Agnostic Learning

In this section, we give a diferentially private algorithm for agnostically learning halfspaces under Gaussian marginals. We build on the polynomial $L ^ { 1 }$ regression algorithm of Kalai et al. (2008), which fits a low-degree polynomial to the labels and then thresholds its predictions.

To privatize this approach, we need two additional properties. We first need a bounded domain that contains the coeficients of a good polynomial approximation to every halfspace. Working in the orthonormal Hermite basis, the unit-norm approximation guarantee of Lemma B.10 allows us to restrict the coeficient vector to the Euclidean unit ball. Second, we need uniformly bounded feature norms so that the regression loss is Lipschitz in the coeficient vector. We obtain this by clipping the feature vectors. This increases the population $L ^ { 1 }$ loss by at most the probability of clipping, which we bound in Lemma C.2. These two properties allow us to fit the coeficients using private convex optimisation (Kulkarni et al., 2021). We then repeat the DP optimisation step on independent samples to boost its success probability and use the exponential mechanism (McSherry and Talwar, 2007) to privately select among the resulting thresholded classifiers. Our main result is stated as follows.

Theorem 2.1 (DP improper agnostic learner for Gaussian marginals). Let $\alpha , \beta , \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } \in ( 0 , 1 / 2 ]$ Set $p = \lceil C \alpha ^ { - 2 } \rceil$ for a suficiently large universal constant $C > 0$ , and $D _ { p } = { \binom { d + p } { p } }$ . Suppose $d \geq C _ { \mathrm { d i m } } \alpha ^ { - 5 }$ , where $C _ { \mathrm { d i m } } > 0$ is a suficiently large universal constant. Let P be any distribution on $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ with marginal $\mu = \mathcal { N } ( 0 , I _ { d } )$ . Then Algorithm 1 is $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) { \cdot } D P$ . Given

$$
n = \widetilde { \mathcal { O } } \left( \frac { D _ { p } } { \alpha ^ { 2 } } + \frac { D _ { p } } { \varepsilon _ { \mathrm { D P } } \alpha } \right)
$$

i.i.d. samples from P, it returns a classifier h satisfying err<sub>P</sub> $( h ) \leq \mathrm { O P T } + \alpha$ with probability at least $1 - \beta$ . The algorithm runs in time poly(n).

The two terms in the sample complexity arise from statistical estimation and private optimization, respectively. When $\varepsilon _ { \mathrm { D P } } \geq \alpha .$ , the privacy term is absorbed into the statistical term, so privacy is free up to logarithmic factors. The sample complexity $\widetilde { \mathcal { O } } \left( D _ { p } / \alpha ^ { 2 } \right)$ matches the dependence on d of known algorithms for non-private agnostic learning of halfspaces under Gaussian marginals (Diakonikolas et al., 2010). Moreover, the $d ^ { 1 / \alpha ^ { 2 } }$ cost is necessary in the statistical query model, even without privacy (Diakonikolas et al., 2021b). In Appendix D, we extend the method to log-concave marginals and establish an analogous guarantee.

## 2.1 The Algorithm

We represent each input x by its feature vector $\psi _ { p } ( x ) = ( h _ { \kappa } ( x ) ) _ { | \kappa | \leq p } \in \mathbb { R } ^ { D _ { p } }$ of normalized Hermite polynomials (O’Donnell, 2014) of degree at most p. The degree $p = \lceil C \alpha ^ { - 2 } \rceil$ sufices to approximate every halfspace in $L ^ { 1 } ( \mu )$ up to error α (Diakonikolas et al., 2010). To bound the influence of any single example and make the regression loss Lipschitz in its coeficients, we clip the feature vectors to norm at most $T = 2 \sqrt { D _ { p } } \colon \bar { \psi } _ { p , T } ( x ) : = T \psi _ { p } ( x ) / \operatorname* { m a x } \{ T , \| \psi _ { p } ( x ) \| _ { 2 } \}$ . Then, by Lemma C.2, the clipping probability is at most $\mathcal O \left( \alpha \right)$ . Given a sample $S ,$ the regression step approximately minimizes the empirical $L ^ { 1 }$ loss

Algorithm 1 Private polynomial $L ^ { 1 }$ regression   
Require: Sample S of size n; parameters $\overline { { \alpha , \beta , \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } } } .$   
1: Set $p = \lceil C \alpha ^ { - 2 } \rceil , T = 2 \sqrt { D _ { p } } , k = \lceil \log _ { 2 } ( 2 / \beta ) \rceil , M = \lceil C / \alpha \rceil$ , and $n _ { 0 } , m$ as in (14b)–(14c).   
2: Partition S into training blocks $S _ { 1 } , \ldots , S _ { k }$ of size $n _ { 0 }$ and a validation block $S _ { \mathrm { v a l } }$ of size m.   
3: for $j = 1 , \dots , k$ do   
4: $\hat { v } _ { j } \gets$ output of the $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ optimizer on $\widehat { L } _ { S _ { j } }$ over $B _ { 2 } ^ { D _ { p } }$   
5: end for   
6: For $( j , \ell ) \in [ k ] \times [ M ]$ , set $t _ { \ell } = - 1 + ( 2 \ell - 1 ) / M$ and $h _ { j , \ell } ( x ) = \mathrm { s i g n } \big ( \langle \hat { v } _ { j } , \bar { \psi } _ { p , T } ( x ) \rangle - t _ { \ell } \big )$   
7: Sample $( \hat { \jmath } , \hat { \ell } )$ with probability proportional to exp $\left( - \varepsilon _ { \mathrm { D P } } m \widehat { \mathrm { e r r } } _ { S _ { \mathrm { v a l } } } ( h _ { j , \ell } ) / 2 \right)$   
8: return $h _ { \hat { j } , \hat { \ell } } .$

$$
\widehat { L } _ { S } ( v ) = \frac { 1 } { | S | } \sum _ { ( X _ { i } , Y _ { i } ) \in S } \left| \langle v , \bar { \psi } _ { p , T } ( X _ { i } ) \rangle - Y _ { i } \right|\tag{4}
$$

over the unit ball $B _ { 2 } ^ { D _ { p } } = \{ v \in \mathbb { R } ^ { D _ { p } } : \| v \| _ { 2 } \leq 1 \}$ . This loss is convex and T-Lipschitz in $v ,$ so we apply the private convex ERM algorithm of Kulkarni et al. (2021), restated in Lemma B.4. A potential alternative approach is to implement polynomial regression using statistical queries (Kearns, 1998) and then simulate these queries privately (Kasiviswanathan et al., 2008). For the clipped regression problem of Equation (4), these queries can estimate the expected subgradients. However, making this SQ implementation explicit and replacing its query answers with actual private estimates leads to an algorithm with basically the same structure as ours. For this reason, we do not write the SQ version of this algorithm here.

Next, the regression guarantee holds in expectation, so we repeat the regression step on independent training blocks to ensure that at least one output has small population loss with high probability. Specifically, Algorithm 1 first partitions the sample into k training blocks of size $n _ { 0 }$ and a validation block of size m. It runs the private optimizer on each training block to obtain coeficient vectors $\hat { v } _ { 1 } , \dotsc , \hat { v } _ { k }$ with scores $\left. \hat { v } _ { j } , \bar { \psi } _ { p , T } ( x ) \right.$ , and then thresholds them at M evenly spaced points in [−1, 1] to produce kM candidate classifiers. Finally, it selects one candidate using the exponential mechanism (McSherry and Talwar, 2007), scoring each candidate by its fraction $\widehat { \mathrm { e r r } } _ { S _ { \mathrm { v a l } } } ( \cdot )$ of misclassified validation examples. Repeating the regression step on k blocks boosts its success probability, and the validation step chooses the coeficient vector and the threshold jointly.

## 2.2 Analysis

In this section, we outline the accuracy analysis; the full accuracy and privacy proofs are deferred to Appendix C. Define the population loss $L _ { P } ( v ) : = \mathbb { E } _ { ( X , Y ) \sim P } { \big | } \langle v , { \bar { \psi } } _ { p , T } ( X ) \rangle - Y { \big | }$ and its minimum over the unit ball $\begin{array} { r } { L ^ { \star } : = \operatorname* { m i n } _ { \| v \| _ { 2 } \leq 1 } L _ { P } ( v ) } \end{array}$ . For a universal constant $c \in ( 0 , 1 / 4 ]$ , we first compare the minimum regression loss with the error of the best halfspace:

$$
L ^ { \star } \leq 2 \ \mathrm { O P T } + 2 c \alpha .\tag{5}
$$

We then show that the algorithm returns a classifier whose error satisfies

$$
\begin{array} { r } { \mathrm { e r r } _ { P } ( h ) \leq \frac { 1 } { 2 } L ^ { \star } + 3 c \alpha } \end{array}\tag{6}
$$

with probability at least $1 - \beta$ . Together, these give err $P ( h ) \leq \mathrm { O P T } + 4 c \alpha \leq \mathrm { O P T } + \alpha$ with the same probability.

We begin with Equation (5), which compares the clipped regression problem with the best halfspace. Fix $f \in { \mathcal { C } }$ . By Lemma B.10 (Diakonikolas et al., 2010; Pesenti et al., 2026), there is a polynomial $q = \langle v _ { q } , \psi _ { p } \rangle$ with $\| v _ { q } \| _ { 2 } \le 1$ and $\| q - f \| _ { L ^ { 1 } ( \mu ) } \leq c \alpha$ . Let $s ( x ) : = T / \operatorname* { m a x } \{ T , \| \psi _ { p } ( x ) \| _ { 2 } \}$ so that $\bar { \psi } _ { p , T } ( x ) = s ( x ) \psi _ { p } ( x )$ . Since $\langle v _ { q } , \bar { \psi } _ { p , T } ( x ) \rangle = s ( x ) \bar { q } ( x )$ with $0 < s ( x ) \leq 1$ , and $| y | = 1$

$$
\begin{array} { c } { { | s ( x ) q ( x ) - y | \leq s ( x ) | q ( x ) - y | + 1 - s ( x ) } } \\ { { \leq | q ( x ) - f ( x ) | + | f ( x ) - y | + 1 - s ( x ) . } } \end{array}
$$

The first inequality follows by adding and subtracting $s ( x ) y$ and applying the triangle inequality. Taking expectations gives

$$
L _ { P } ( v _ { q } ) \leq 2 \operatorname { e r r } _ { P } ( f ) + \| q - f \| _ { L ^ { 1 } ( \mu ) } + \mathbb { E } _ { \mu } [ 1 - s ( X ) ] .
$$

The three terms are the classification error of $f ,$ the polynomial approximation error, and the clipping bias. Since $1 - s ( x ) \leq \mathbf { 1 } \{ \| \psi _ { p } ( x ) \| _ { 2 } > T \}$ , Lemma C.2 bounds the last term by cα. Clipping thus enters only through its probability. The approximation error is also at most $c \alpha ,$ so $L _ { P } ( v _ { q } ) \le 2 \mathrm { e r r } _ { P } ( f ) + 2 c \alpha$ As $\| v _ { q } \| _ { 2 } \leq 1$ , we have $L ^ { \star } \leq L _ { P } ( v _ { q } )$ , and taking the infimum over $f$ gives (5).

For Equation (6), we first show that at least one of the k regression runs is nearly optimal. The loss is convex and T-Lipschitz and thus for each training block, Lemma B.4 from Kulkarni et al. (2021) bounds the expected optimization error. Then, standard Rademacher complexity bound for linear predictors under $L _ { 1 }$ loss (Lemma B.7), applied with $\mathbb { E } _ { \mu } \| \bar { \psi } _ { p , T } ( X ) \| _ { 2 } ^ { 2 } \leq D _ { p }$ , bounds the expected deviation between the empirical and population losses. We choose n so that these two contributions sum to at most cα (see (14b)), giving

$$
\mathbb { E } \big [ L _ { P } ( \hat { v } _ { j } ) \big ] - L ^ { \star } \leq c \alpha ,
$$

where the expectation is over $S _ { j }$ and the optimizer’s randomness. Since $\hat { v } _ { j }$ lies in the unit ball, its excess loss $L _ { P } ( \hat { v } _ { j } ) - L ^ { \star }$ is non-negative. Markov’s inequality therefore gives $L _ { P } ( \hat { v } _ { j } ) \leq L ^ { \star } + 2 c \alpha$ with probability at least $1 / 2$ . The k runs are independent, so with probability at least $1 - 2 ^ { - k } \geq 1 - \beta / 2$ some run $j ^ { \star }$ satisfies this bound.

We next show that thresholding this run yields a good candidate classifier. For any fixed regression output $\hat { v } _ { j }$ , a uniformly random threshold $\tau \in [ - 1 , 1 ]$ gives expected classification error at most $L _ { P } ( \hat { v } _ { j } ) / 2$ , where the expectation is over τ (Kalai et al., 2008, see also Fact B.6). Rounding $\tau$ to the midpoint of its grid interval makes it uniform over the M grid thresholds and increases the expected classification error by at most $1 / ( 2 M ) \leq c \alpha$ . Thus, at least one grid threshold has error no larger than this average. On the event that some run $j ^ { \star }$ satisfies this loss bound, we have that

$$
\begin{array} { r } { \underset { j , \ell } { \operatorname* { m i n } } \operatorname { e r r } _ { P } ( h _ { j , \ell } ) \leq \frac { 1 } { 2 } L _ { P } ( \widehat { v } _ { j ^ { \star } } ) + c \alpha \leq \frac { 1 } { 2 } L ^ { \star } + 2 c \alpha . } \end{array}
$$

This establishes that the candidate set contains a good classifier. It remains to select one privately. The algorithm uses the exponential mechanism on the independent validation sample $S _ { \mathrm { v a l } }$ to select among the kM candidate classifiers. By Lemma B.5, with m as in (14c), the selected classifier has error at most cα above the best candidate with probability at least $1 - \beta / 2$ . A union bound over the training and validation failure events proves Equation (6).

Finally, since $T \sqrt { D _ { p } } = 2 D _ { p }$ and $k = \mathcal { O } \left( \log ( 1 / \beta ) \right)$ , the choices of $n _ { 0 }$ and m give the sample bound of Theorem 2.1. Computing the features, running the k private optimizers, and evaluating the kM candidates on $S _ { \mathrm { v a l } }$ take $\mathrm { p o l y } ( n )$ arithmetic operations. The detailed proof, along with privacy and runtime calculations appear in Appendix C.

Remark 3. The dimension assumption in Theorem 2.1 allows us to use the clipping radius $T = 2 \sqrt { D _ { p } }$ To obtain a guarantee in every dimension, we instead take $T = \sqrt { D _ { p } / ( c \alpha ) }$ for a suficiently small universal constant $c > 0$ . Orthonormality and Markov’s inequality give

$$
\operatorname* { P r } _ { \mu } ( \| \psi _ { p } ( X ) \| _ { 2 } > T ) \leq \frac { D _ { p } } { T ^ { 2 } } = c \alpha .\tag{7}
$$

Replacing Lemma C.2 with this bound, the same analysis (see Appendix C) establishes the learning guarantee of Theorem 2.1 for every $d \geq 1$ . The statistical term in the sample complexity remains unchanged, and the privacy term acquires an additional α<sup>−1/2</sup> factor. $- 1 / 2$

## 3 Private Proper Agnostic Learning

The agnostic learner in Section 2 is improper since its output need not be a halfspace. In this section, we provide a private proper learner that outputs a halfspace in $\mathbb { R } ^ { d }$ . A halfspace requires only $\mathcal O \left( d \right)$ parameters, compared to $d ^ { \mathcal { O } \left( 1 / \alpha ^ { 2 } \right) }$ coeficients for the improper hypothesis, and is faster to evaluate on a point and also easier to interpret.

We will adapt the low-rank subspace reduction of Diakonikolas et al. (2021a) to design the learner. First, it constructs a finite class of halfspaces from a polynomial with suficiently small excess squared loss. The resulting class is guaranteed to contain a halfspace whose error is close to the best halfspace. Once we have this finite class of halfspaces, we use the exponential mechanism (McSherry and Talwar, 2007) to privately select a halfspace from it. Our main result is stated below.

Theorem 3.1 (DP proper agnostic learner). Let $\alpha , \beta , \delta _ { \mathrm { D P } } ~ \in ~ ( 0 , 1 / 2 )$ and $\varepsilon _ { \mathrm { D P } } ~ \in ~ ( 0 , 1 )$ Set $p = \lceil C _ { 0 } / ( c \alpha ) ^ { 2 } \rceil$ for a suficiently large universal constant $C _ { 0 } > 0$ and a suficiently small universal constant $c > 0$ , let $D _ { p } = { \binom { d + p } { p } }$ , and suppose that $d \geq C _ { \mathrm { d i m } } \alpha ^ { - 5 }$ for a suficiently large universal constant $C _ { \mathrm { d i m } } > 0$ . Let P be any distribution on $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ with marginal $\mu = \mathcal { N } ( 0 , I _ { d } )$ . Then Algorithm 2 is $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) { - } D P$ and, given

$$
n = \widetilde { \mathcal { O } } \left( \frac { D _ { p } } { \alpha ^ { 2 } } + \frac { D _ { p } } { \varepsilon _ { \mathrm { D P } } \alpha } + \frac { 1 } { \alpha ^ { 6 } } + \frac { 1 } { \varepsilon _ { \mathrm { D P } } \alpha ^ { 5 } } \right)
$$

i.i.d. samples from P, returns a halfspace $h \in { \mathcal { C } }$ satisfying err $P ( h ) \leq \mathrm { O P T } + \alpha$ with probability at least $1 - \beta$ . The algorithm runs in time poly $\cdot ( n ) \cdot \exp \bigl ( O ( \alpha ^ { - 4 } \log ( 1 / \alpha ) ) \bigr )$

A proof sketch appears at the end of this section, and the full proof is given in Appendix E. We use a strengthened version of the subspace reduction of Diakonikolas et al. (2021a), reducing the polynomial degree from $\mathcal { O } \left( 1 / \alpha ^ { 4 } \right)$ to $\mathcal { O } \left( 1 / \alpha ^ { 2 } \right)$ . This reduces the subspace dimension from $\mathcal { O } \left( 1 / \alpha ^ { 6 } \right)$ to $\mathcal { O } \left( 1 / \alpha ^ { 4 } \right)$ and yields a smaller finite class for the exponential mechanism. We also extend the reduction to labels in $[ - 1 , 1 ]$ . Lemma 3.2 states this guarantee.

Thus, our sample bound has the same $\textstyle { \binom { d + p } { p } }$ dependence on the dimension as non-private improper agnostic learning under Gaussian marginals (Diakonikolas et al., 2010), while returning a halfspace. When $\varepsilon _ { \mathrm { D P } } \geq \alpha$ , the privacy terms are absorbed into the statistical terms $D _ { p } / \alpha ^ { 2 } + 1 / \alpha ^ { 6 }$ , up to logarithmic factors. In this regime, privacy comes essentially for free.

The factor $\exp \bigl ( \mathcal O \left( \alpha ^ { - 4 } \log ( 1 / \alpha ) \right) \bigr )$ in the running time arises from constructing and searching a finite cover of halfspaces of at most this size, so the running time is polynomial in n for every fixed accuracy α. The non-private proper learner of Diakonikolas et al. (2021a) also searches such a cover, and its running time likewise includes a term exponential in $\mathrm { p o l y } ( 1 / \alpha )$

Lemma 3.2. There exist universal constants $C _ { 0 } , C , c _ { 0 } > 0$ such that the following holds. Let $( X , U )$ have any joint distribution with $X \sim { \mathcal { N } } ( 0 , I _ { d } )$ and $| U | \le 1$ . Fix $\alpha \in ( 0 , 1 / 2 )$ , and set $p = \left\lceil C _ { 0 } \alpha ^ { - 2 } \right\rceil$ Let $\mathcal { P } _ { p }$ denote the space of polynomials on $\mathbb { R } ^ { d }$ of degree at most $p .$ . Suppose that $q \in \mathcal { P } _ { p }$ satisfies

$$
\mathbb { E } \left[ ( U - q ( X ) ) ^ { 2 } \right] \leq \operatorname* { i n f } _ { Q \in { \mathcal { P } } _ { p } } \mathbb { E } \left[ ( U - Q ( X ) ) ^ { 2 } \right] + c _ { 0 } ^ { 2 } \alpha ^ { 2 } .\tag{8}
$$

Using just q and α, we can construct a finite class of halfspaces $\mathcal { H } ( q ) \subseteq \mathcal { C }$ satisfying log $| \mathcal { H } ( q ) | \leq$ $C \alpha ^ { - 4 } \log ( 1 / \alpha )$ , such that, for every $f \in { \mathcal { C } }$ , there exists $h \in \mathcal { H } ( q )$ with E $[ U \left( f ( X ) - h ( X ) \right) ] \leq C \alpha$

The proof is stated in Appendix E. A natural idea is to obtain the polynomial $q$ from our private improper learner in Theorem 2.1. However, the coeficients $\widehat { v } _ { j }$ obtained in each regression run of Algorithm 1 only satisfy the following $L ^ { 1 }$ guarantee:

$$
\mathbb { E } _ { ( X , Y ) \sim P } \left| \left. \hat { v } _ { j } , \bar { \psi } _ { p , T } ( X ) \right. - Y \right| \leq \operatorname* { m i n } _ { \| v \| _ { 2 } \leq 1 } \mathbb { E } _ { ( X , Y ) \sim P } \left| \left. v , \bar { \psi } _ { p , T } ( X ) \right. - Y \right| + \mathcal { O } \left( \alpha \right) ,
$$

which does not imply the excess squared-loss bound in Equation (8). Moreover, the function $\big \langle \widehat { v } _ { j } , \bar { \psi } _ { p , T } ( X ) \big \rangle$ is a linear combination of clipped Hermite features $\bar { \psi } _ { p , T } ( X )$ and therefore need not be a polynomial as required in Lemma 3.2. We therefore need a diferent private polynomial construction that directly controls excess squared loss.

## 3.1 Constructing a private polynomial with small excess squared loss

To construct the polynomial, we propose a simple estimator based on the correlations between the labels and the Hermite features. For squared-loss regression, these correlations give the coeficients of the optimal polynomial (see Lemma 3.3 below). However, clipping changes the correlations between the labels and the Hermite features. Write $m = \mathbb { E } \left[ Y \psi _ { p } ( X ) \right]$ for the original correlation vector and $m _ { T } = \mathbb { E } \left[ Y \bar { \psi } _ { p , T } ( X ) \right]$ for the correlation vector computed using clipped features. Then

$$
\begin{array} { r l r } {  { \| m - m _ { T } \| _ { 2 } = \operatorname* { s u p } _ { \| v \| _ { 2 } = 1 }  \mathbb { E } [ Y ( 1 - s ( X ) ) \langle v , \psi _ { p } ( X ) \rangle ]  } } \\ & { } & \\ & { } & { \leq \sqrt { \mathbb { E } [ ( 1 - s ( X ) ) ^ { 2 } ] } \leq \sqrt { \operatorname* { P r } _ { \mu } [ \| \psi _ { p } ( X ) \| _ { 2 } > T ] } = \mathcal { O } ( \sqrt { \alpha } ) , } \end{array}
$$

where the first inequality uses Cauchy–Schwarz and orthonormality of Hermite functions, and the last inequality uses $\mathcal O \left( \alpha \right)$ bound on the clipping probability from Section 2 (see Lemma C.2). Thus, a $\mathcal { O } \left( \sqrt { \alpha } \right)$ bound on the diference between the original and clipped correlations only gives an $\mathcal O \left( \alpha \right)$ bound on excess squared loss, whereas Lemma 3.2 requires an $\mathcal { O } \left( \alpha ^ { 2 } \right)$ excess squared loss. Therefore, we need our estimated coeficients to be within $\mathcal O \left( \alpha \right)$ of the optimal coeficients.

We show that this can be achieved by constructing the polynomial for the clipped labels $U = s ( X ) Y$ , using the loss $L _ { s } ( Q ) = \operatorname { \mathbb { E } } \left\lceil ( U - Q ( X ) ) ^ { 2 } \right\rceil$ . Note that $| U | = | s ( X ) Y | \le 1$ . Then, the corresponding correlation vector is $m _ { T } \ \overset { \cdot } { = } \mathbb { E } \left[ Y \bar { \psi } _ { p , T } ( \bar { X } ) \right] = \mathbb { E } \left[ U \psi _ { p } ( X ) \right]$ . By orthonormality of the Hermite basis, m<sub>T</sub> is exactly the coeficient vector minimizing $L _ { s }$ over $\mathcal { P } _ { p }$

Lemma 3.3. Let $X \sim \mathcal { N } ( 0 , I _ { d } )$ , let $| U | \le 1$ , and write $m _ { T } = \mathbb { E } \left[ U \psi _ { p } ( X ) \right]$ . The polynomial $q _ { m _ { T } } ( x ) = \langle m _ { T } , \psi _ { p } ( x ) \rangle$ is the unique minimizer of $L _ { s }$ over $\mathcal { P } _ { p }$

Proof of Lemma 3.3. Every polynomial in ${ \mathcal P _ { p } }$ can be written as $q _ { w } ( x ) = \langle w , \psi _ { p } ( x ) \rangle$ ⟩ for some $w \in \mathbb { R } ^ { D _ { p } }$ Using orthonormality of the Hermite basis, we get E $\left[ \psi _ { p } ( X ) \psi _ { p } ( X ) ^ { \top } \right] = I _ { D _ { p } }$ . Then,

$$
\begin{array} { r l } & { L _ { s } ( q _ { w } ) = \mathbb { E } \left[ \left( U - q _ { w } ( X ) \right) ^ { 2 } \right] = \mathbb { E } \left[ U ^ { 2 } \right] - 2 \left. w , m _ { T } \right. + \left. w \right. _ { 2 } ^ { 2 } } \\ & { \qquad = \mathbb { E } \left[ U ^ { 2 } \right] - \left. m _ { T } \right. _ { 2 } ^ { 2 } + \left. w - m _ { T } \right. _ { 2 } ^ { 2 } . } \end{array}\tag{9}
$$

```latex
Algorithm 2 Private proper learning of Gaussian halfspaces
Require: Sample S of size n, accuracy α, failure probability $\beta ,$ and privacy parameters $\left( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } \right)$
1: Set $p = \tilde { \lceil C _ { 0 } / ( c \alpha ) ^ { 2 } \rceil } , D _ { p } = \binom { d + p } { p }$ , and $T = 2 { \sqrt { D _ { p } } } ;$ , where $c > 0$ is a suficiently small universal
constant.
2: Split S into disjoint blocks $S _ { 1 } , S _ { 2 }$ of size $n / 2$ each, independently of the sample values.
3: Draw $Z \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { D _ { p } } )$ with σ $\sigma = \frac { 4 T \sqrt { 2 \log ( 1 . 2 5 / \delta _ { \mathrm { D P } } ) } } { \ l }$
ε<sub>DP</sub>n
4: $\widetilde { m }  \mathrm { P r o j } _ { B _ { 2 } ^ { D _ { p } } } ( \frac { 2 } { n } \sum _ { ( X _ { i } , Y _ { i } ) \in S _ { 1 } } Y _ { i } \bar { \psi } _ { p , T } ( X _ { i } ) + Z )$ and set $q ( x ) \gets \langle \widetilde { m } , \psi _ { p } ( x ) \rangle$
5: Construct $\mathcal { H } ( q )$ using the construction in Lemma 3.2 with accuracy cα.
6: Sample $h \in \mathcal { H } ( q )$ with probability proportional to exp $\left( - ( \varepsilon _ { \mathrm { D P } } / 2 ) \sum _ { ( X _ { i } , Y _ { i } ) \in S _ { 2 } } \mathbf { 1 } \left\{ h ( X _ { i } ) \neq Y _ { i } \right\} \right)$
7: return h.
```

The first two terms do not depend on w. Thus, the loss is minimized uniquely at $w = m _ { T }$ □

Thus, our goal is to estimate m<sub>T</sub> privately. For any estimate $\widetilde { m } .$ , using Equation (9), we get that the polynomial $q _ { \widetilde { m } } ( x ) = \langle \widetilde { m } , \psi _ { p } ( x ) \rangle$ satisfies

$$
L _ { s } ( q _ { \widetilde m } ) - \operatorname* { i n f } _ { Q \in \mathcal { P } _ { p } } L _ { s } ( Q ) = \left. \widetilde m - m _ { T } \right. _ { 2 } ^ { 2 } .
$$

It therefore sufices to estimate a coeficient vector $\widetilde { m }$ such that $\begin{array} { r } { \| \widetilde { \boldsymbol { m } } - \boldsymbol { m } _ { T } \| _ { 2 } \leq } \end{array}$ c<sub>0</sub>cα for some constant $c _ { 0 } > 0$ . The regression step in Algorithm $2 \ ( \mathrm { S t e p \ 4 } )$ estimates these correlations using the Gaussian mechanism and projects the estimate onto the coeficient unit ball. Since $\| m _ { T } \| _ { 2 } \leq 1$ , projection cannot increase the estimation error. The following lemma gives the guarantee for the resulting polynomial.

Lemma 3.4. Let $\alpha , \beta , \delta _ { \mathrm { D P } } \in ( 0 , 1 / 2 )$ and $\varepsilon _ { \mathrm { D P } } \in ( 0 , 1 ]$ . Let q be the polynomial constructed in Step 4 of Algorithm 2, using the $n / 2$ independent samples in $S _ { 1 }$ . Its release is $\big ( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } \big )$ -diferentially private, and $\| q \| _ { L ^ { 2 } ( \mu ) } \leq 1$ on every run. If $n / 2 = \widetilde \Omega \left( D _ { p } / \alpha ^ { 2 } + D _ { p } / ( \varepsilon _ { \mathrm { D P } } \alpha ) \right)$ , then, with probability at least $1 - \beta / 2 ,$

$$
L _ { s } ( q ) \leq \operatorname* { i n f } _ { Q \in \mathcal { P } _ { p } } L _ { s } ( Q ) + c _ { 0 } ^ { 2 } c ^ { 2 } \alpha ^ { 2 } .\tag{10}
$$

The proof is given in Appendix E.

## 3.2 Constructing the finite set of halfspaces

We now use the polynomial q computed in Step 4 of Algorithm 2 to construct a finite class of halfspaces. On the event in Lemma 3.4, this polynomial satisfies the squared-loss condition (Equation (8)) of Lemma 3.2 with accuracy cα. We therefore apply Lemma 3.2 in Step 5 to construct $\mathcal { H } ( q )$ . The full algorithm and proof are given in Appendix E. The algorithm follows the subspace and discretization construction of Diakonikolas et al. (2021a). Specifically, we compute

$$
M = \mathbb { E } _ { X \sim \mathcal { N } ( 0 , I _ { d } ) } \left[ \nabla q ( X ) \nabla q ( X ) ^ { \top } \right]
$$

and take V to be the span of its eigenvectors with eigenvalues at least $c ^ { 2 } \alpha ^ { 2 } / C _ { 0 }$ . They show that the resulting subspace has dimension $\mathcal { O } \left( \alpha ^ { - 4 } \right)$ . Discretizing unit normal vectors in V gives a finite

class of halfspaces satisfying log $\vert \mathcal { H } ( q ) \vert = \mathcal { O } \left( \alpha ^ { - 4 } \log ( 1 / \alpha ) \right)$ . All these operations use only q and the known Gaussian marginal, so Algorithm 2 preserves DP by post-processing (Lemma B.3).

Lemma 3.4 guarantees the polynomial q has small excess squared loss for the clipped labels U. Lemma 3.5 next translates this guarantee into classification error for the original labels.

Lemma 3.5. Suppose that $d \geq C _ { \mathrm { d i m } } \alpha ^ { - 5 }$ for a suficiently large universal constant $C _ { \mathrm { d i m } }$ and C is the constant in Lemma 3.2. Let q and $\mathcal { H } ( q )$ be constructed by Algorithm 2. If q satisfies Equation (10),

$$
\operatorname* { m i n } _ { h \in \mathcal H ( q ) } \mathrm { e r r } _ { P } ( h ) \le \mathrm { O P T } + ( C / 2 + 1 ) c \alpha .
$$

Proof. On the event in the lemma, Lemma 3.2 gives, for every halfspace $f \in { \mathcal { C } } ,$ , an $h \in \mathcal { H } ( q )$ satisfying E $[ U \left( f ( X ) - h ( X ) \right) ] \leq C c \alpha$ . Since

$$
\begin{array} { r } { \operatorname { e r r } _ { P } ( h ) - \operatorname { e r r } _ { P } ( f ) = \frac { 1 } { 2 } \mathbb { E } \left[ Y \left( f ( X ) - h ( X ) \right) \right] \leq \frac { 1 } { 2 } \mathbb { E } \left[ U \left( f ( X ) - h ( X ) \right) \right] + \mathbb { E } \left[ 1 - s ( X ) \right] , } \end{array}
$$

it remains to bound the last term. As $s ( X ) = 1$ outside the clipping event, Lemma C.2 gives $\mathbb { E } \left[ 1 - s ( X ) \right] \leq \mathbb { P } \left[ \left\| \psi _ { p } ( X ) \right\| _ { 2 } > T \right] \leq$ cα when $T = 2 \sqrt { D _ { p } }$ and $d \geq 2 p ^ { 2 } / ( c \alpha )$ □

Thus, $\mathcal { H } ( q )$ contains a halfspace with error at most $\mathrm { O P T } + ( C / 2 + 1 ) c \alpha$

## 3.3 Selecting the best halfspace

Finally, Algorithm 2 selects a halfspace from $\mathcal { H } ( q )$ using the exponential mechanism, with scores equal to the negative empirical errors on $S _ { 2 }$ . These errors are measured against the original labels Y. Conditional on the released polynomial, $\mathcal { H } ( q )$ is fixed and independent of $S _ { 2 }$ . By Lemma B.5, this selection adds at most cα error with probability at least $1 - \beta / 2$ , provided

$$
n = \Omega \left( \left( \alpha ^ { - 4 } \log ( 1 / \alpha ) + \log ( 1 / \beta ) \right) \left( { \frac { 1 } { \alpha ^ { 2 } } } + { \frac { 1 } { \varepsilon _ { \mathrm { D P } } \alpha } } \right) \right) .\tag{11}
$$

Combining this with Lemma 3.4 and taking c suficiently small gives error at most $\mathrm { O P T } + \alpha$ , with probability at least $1 - \beta$ . The private regression step uses $S _ { 1 }$ , while the exponential mechanism uses the disjoint block S<sub>2</sub>, each of size $n / 2$ . Therefore, post-processing and parallel composition give $\left( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } \right)$ -privacy for Algorithm 2.

Proof Sketch of Theorem 3.1. Algorithm 2 is the proper private learner achieving the guarantee in Theorem 3.1. Its output belongs to $\mathcal { H } ( q ) \subseteq \mathcal { C }$ , so it is proper. Lemma 3.4 guarantees that Algorithm 2 outputs a polynomial satisfying Equation (8) in Lemma 3.2 with accuracy cα, with probability at least $1 - \beta / 2$ . Thus, Step 5 constructs $\mathcal { H } ( q )$ from a subspace of dimension $\mathcal { O } \left( \alpha ^ { - 4 } \right)$ . Lemma 3.5 ensures that it contains a halfspace with error at most $\mathrm { O P T } + ( C / 2 + 1 ) c \alpha$ . When the sample size n satisfies Equation (11), the exponential mechanism adds at most cα error with probability at least $1 - \beta / 2$ by Lemma B.5; a union bound and suficiently small c give the claimed accuracy. Adding the validation sample cost to the sample cost in Lemma 3.4 gives the stated sample complexity.

For privacy, note that Step 4 is $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ by Lemma 3.4, Step 5 is post-processing, and Step 6 is $\varepsilon _ { \mathrm { D P ^ { - } } } \mathrm { D P }$ on $S _ { 2 }$ by exponential mechanism, conditioned on the released polynomial. Since $S _ { 1 }$ and $S _ { 2 }$ are disjoint, parallel composition makes the entire algorithm $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ □

## 4 Private Proper Semi-Agnostic Learning

In this section, we provide a DP algorithm for proper semi-agnostic learning of halfspaces whose sample complexity is polynomial in d and $1 / \alpha$ , rather than exponential in $1 / \alpha$ as in Theorems 2.1 and 3.1. The diference between semi-agnostic and agnostic error guarantee is that the former allows a universal constant factor multiplying OPT. We use a privatised version of the non-private gradient descent (on a non-convex function) algorithm of Diakonikolas et al. (2022) as a subroutine. Roughly speaking, given an upper bound $\eta \geq \mathrm { O P T }$ , this private subroutine (Algorithm 3) returns a halfspace with error $\mathcal { O } \left( \eta \right)$ . Then, our main algorithm (Algorithm 4) searches over a small set of candidate values of $\eta$ and selects a hypothesis using the exponential mechanism. We state our main result below and defer the privacy proof and technical details of the accuracy analysis to Appendix F.

Theorem 4.1 (DP semi-agnostic proper learner). Let $\alpha , \beta , \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } \in ( 0 , 1 )$ . Let P be any distribution on $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ with marginal $\mu = \mathcal { N } ( 0 , I _ { d } )$ . Then Algorithm $\it 4$ is $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) { \cdot } D P$ and, given

$$
n = \widetilde { \mathcal { O } } \left( \frac { d } { \alpha ^ { 2 } } + \frac { d } { \varepsilon _ { \mathrm { D P } } \alpha } \right)
$$

$i . i . d .$ samples from $P ,$ , where $d \ge 1 6 \log ( n / \beta )$ , it returns a halfspace $h \in { \mathcal { C } }$ satisfying err $\cdot _ { P } ( h ) \leq$ $C \cdot \mathrm { O P T } + \alpha$ with probability at least $1 - \beta$ , where $C > 0$ is a universal constant. The algorithm runs in time poly(n).

We make the above assumption on d (Appendix F.2) for ease of exposition. It can be removed by setting the clipping parameter L in Algorithm 3 to a tighter bound of $O ( \sqrt { d } + \sqrt { \log ( n / \beta ) } )$ , thus simply introducing more polylog terms in the sample complexity that we already omit in our presentation.

## 4.1 The Algorithm

We first describe the subroutine in Algorithm 3, which takes a known upper bound $\eta \geq \mathrm { O P T }$ and performs gradient updates using a ramp loss. For $a > 0$ and $b \in \mathbb { R }$ , define

$$
r _ { a , b } ( z ) = { \left\{ \begin{array} { l l } { 0 } & { { \mathrm { i f ~ } } z \leq b - a , } \\ { z - b + a } & { { \mathrm { i f ~ } } b - a < z < b + a , } \\ { 2 a } & { { \mathrm { i f ~ } } z \geq b + a . } \end{array} \right. }\tag{12}
$$

Algorithm 3 follows Algorithms 1 and 2 from Diakonikolas et al. (2022), with noise added to the computations that access the dataset. The algorithm needs to privately estimate two quantities from the dataset before running gradient descent: the threshold parameter $\tau$ and the initial weight vector $\hat { w } _ { 0 }$ . Diakonikolas et al. (2022) first estimate the threshold from the average of the labels in $D _ { \tau }$ , using the inverse Gaussian CDF. We privatise this average by adding Laplace noise, obtaining the threshold estimate $\hat { \tau }$ in Algorithm 3. Next, we compute the initial weight vector $\hat { w } _ { 0 }$ by adding Gaussian noise to the average of xy over $D _ { i n i t }$ in Algorithm 3. The algorithm then keeps $\hat { \tau }$ fixed, and performs gradient descent using the ramp function in Equation (12) and the private threshold $\hat { \tau } _ { ; }$ with a fresh block $D _ { t }$ in each iteration. Algorithm 3 adds Gaussian noise to each empirical gradient before performing the gradient update.

With these modifications, Algorithm 3 privately returns a hypothesis with population classification error $\mathcal { O } \left( \eta \right)$ whenever $\eta \geq \mathrm { O P T }$ , as shown in Proposition 4.2. Since OPT is unknown, Algorithm 4 tries a geometric grid of candidate values $\eta _ { 1 } , \dots , \eta _ { K }$ . For each $\eta _ { k }$ , it runs Algorithm 3 on a separate subset of the data to obtain a hypothesis $h _ { k }$ . The grid is chosen such that at least one of these hypotheses has error $\mathcal { O } \left( \mathrm { O P T } + \alpha \right)$ with high probability. The algorithm then uses the exponential mechanism on a separate validation set $D _ { v a l }$ to select a hypothesis whose empirical classification error is close to the smallest among these candidates.

Algorithm 3 DP Semi-Agnostic Subroutine DPSAS   
Require: Dataset D of $n _ { 0 }$ labelled examples, privacy parameters $\varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } > 0 .$ , and error parameter   
η.   
Partition Dataset and Clip for DP   
1: $s  \Theta ( 1 / \log ^ { 3 / 2 } ( 1 / \eta ) )$ and $T \gets \Theta ( \log ^ { 4 } ( 1 / \eta ) )$   
2: Partition $D$ into $D _ { i n i t } , D _ { \tau } , D _ { 0 } , \dots , D _ { T - 1 } ,$ , each of size $m _ { 0 } \gets | D | / ( T + 2 )$   
3: $L \gets 2 \sqrt { d }$ and clip each feature vector in D to have norm at most $L .$   
Privately computing threshold τˆ.   
4: $\kappa  C \eta$ for a universal constant $C$ and $\hat { y }  \mathbb { E } _ { ( x , y ) \sim D _ { \tau } } [ y ] + z _ { \tau } .$   
where $\begin{array} { r } { z _ { \tau } \sim \mathsf { L a p } \left( \frac { 2 } { \varepsilon _ { \mathrm { D P } } m _ { 0 } } \right) } \end{array}$   
5: $\hat { \tau } \gets \Phi ^ { - 1 } \left( \mathsf { c l i p } \left[ \frac { 1 + \hat { y } } { 2 } \right] _ { \kappa } ^ { 1 - \kappa } \right)$   
Privately computing initial weight $\hat { w } _ { 0 } .$   
6: $\tilde { w } _ { 0 } \gets \mathbb { E } _ { ( x , y ) \sim D _ { i n i t } } [ x y ] + z _ { 0 } ,$   
where $z _ { 0 } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } \mathbb { I } )$ and $\begin{array} { r } { \sigma \gets \frac { 2 L \sqrt { 2 \log ( 2 / \delta _ { \mathrm { D P } } ) } } { \sigma - \infty } } \end{array}$ ε<sub>DP</sub>m<sub>0</sub>   
7: $\begin{array} { r } { \hat { w } _ { 0 } \gets \frac { \tilde { w } _ { 0 } } { \lVert \tilde { w } _ { 0 } \rVert } \mathrm { ~ i f ~ } \lVert \tilde { w } _ { 0 } \rVert \neq 0 . } \end{array}$ , otherwise $\hat { w } _ { 0 } \sim \mathcal { S } ^ { d - 1 }$ uniformly at random.   
Run DP online gradient descent.   
8: $\lambda  e ^ { \hat { \tau } ^ { 2 } / 2 } s$ and $B \gets S ^ { d - 1 }$   
9: for $t = 0$ to $T - 1$ do   
10: ϕ<sub>t</sub> $ ( \pi / 2 ) ( 1 - s ^ { 2 } / \underline { { 6 } } 4 ) ^ { t } , a _ { t }  \sin ( \phi _ { t } )$ , and $b _ { t } \gets - \hat { \tau } \cos ( \phi _ { t } )$   
11: $\hat { \mathcal { L } } _ { t } ( w ) \gets \mathbb { E } _ { ( x , y ) \sim D _ { t } } \left[ - r _ { a _ { t } , b _ { t } } \left( \frac { w \cdot x } { \| w \| } \right) y \right]$   
12: $\hat { u } \gets \hat { w } _ { t } - \lambda \big ( \nabla \hat { \mathcal { L } } _ { t } ( \hat { w } _ { t } ) + z _ { t + 1 } \big )$ and $\hat { w } _ { t + 1 } \gets \mathrm { p r o j } _ { B } ( \hat { u } )$   
where $z _ { t + 1 } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ( \mathbb { I } - \hat { w } _ { t } \hat { w } _ { t } ^ { \top } ) )$   
13: end for   
14: Return ${ \hat { w } } _ { T } , { \hat { \tau } } .$

## 4.2 Accuracy

A major component for proving the utility for Algorithm 4 is proving the accuracy guarantees for Algorithm 3. Note that by construction, $\alpha \leq \eta _ { k }$ for all $k \in [ K ]$ . In the presentation for the rest of Section 4.2, we will omit the subscript from $\eta _ { k }$ to refer to the parameter η in Algorithm 3. We write m for the number of examples in each of the $T + 2$ disjoint blocks used by the algorithm; the total sample size for one run is $n _ { 0 } = ( T + 2 ) m _ { 0 }$

Proposition 4.2. Let $\eta \in ( 0 , 1 / 2 )$ and $\beta , \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } \in ( 0 , 1 )$ . Let P be any distribution on $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ with marginal $\mu = \mathcal { N } ( 0 , I _ { d } )$ and $\mathrm { O P T } \le \eta$ . Then Algorithm 3 is $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ and, given

$$
m _ { 0 } = \widetilde { \mathcal { O } } \left( \frac { d } { \eta ^ { 2 } } + \frac { d } { \varepsilon _ { \mathrm { D P } } \eta } \right)
$$

i.i.d. samples from P in each of its blocks, where $d \ge 1 6 \log ( n _ { 0 } / \beta )$ for the total sample size $n _ { 0 }$ returns a halfspace $h \in { \mathcal { C } }$ satisfying $\mathrm { e r r } _ { P } ( h ) \leq C \eta$ with probability at least $1 - \beta$ , where $C > 0$ is a universal constant.

We follow the utility analysis of Diakonikolas et al. (2022), with modifications to account for the noise added for privacy. Their analysis starts with a weight vector that makes an angle less than $\pi / 2$ with the optimal halfspace $w ^ { \star }$ . In each iteration, the gradient update reduces this angle until the weight vector is suficiently close to $w ^ { \star }$ to give the desired error guarantee. We first analyse these updates assuming that our private initialization satisfies this starting condition and that replacing the optimal threshold by our private estimate preserves a small classification and then prove it in Claims 4.5 and 4.6. They first prove population-level results, where they show that applying population-level gradients will slowly lead us to $w ^ { \star }$ , and then use concentration bounds to show that the empirical gradients satisfy the estimates needed for convergence with high probability. We modify these empirical estimates to account for the additional noise added for privacy.

The first result describes the structure of the noisy gradient and the updated weight vector. Its proof is given in Appendix F.

Claim 4.3 (Modification of Claim C.2 from Diakonikolas et al. (2022)). For any unit vector $w \in \mathbb { R } ^ { d }$ the DP gradient rule from Algorithm 3 with loss $\hat { \mathcal { L } } _ { t } ( w )$ and stepsize λ corresponds to the update:

$$
w ^ { \prime } = \frac { w - \lambda \nabla _ { w } \tilde { \mathcal { L } } _ { t } ( w ) } { \Vert w - \lambda \nabla _ { w } \tilde { \mathcal { L } } _ { t } ( w ) \Vert } ,
$$

such that $\begin{array} { r } { \nabla _ { w } \bar { \mathcal { L } } _ { t } ( w ) = \mathbb { E } _ { ( x , y ) \sim D _ { t } } \left[ - y \mathbf { 1 } \{ | w \cdot x - b _ { t } | \leq a _ { t } \} \nabla _ { w } \frac { w \cdot x } { \| w \| } \right] + z _ { t + 1 } , \mathrm { w h e r e } ~ z _ { t + 1 } \sim \mathcal { N } \left( 0 , ( \mathbb { I } - w w ^ { \top } ) \sigma ^ { 2 } \right) } \end{array}$ and $\begin{array} { r } { \sigma = \frac { 2 L \sqrt { 2 \log ( 2 / \delta _ { \mathrm { D P } } ) } } { \varepsilon _ { \mathrm { D P } } m _ { 0 } } } \end{array}$

Next, we prove the DP analogue of Lemma C.12 from Diakonikolas et al. (2022) and show that the original additive error guarantees are not afected by the added noise for DP. We keep the notation consistent with theirs, and define $g ( w , \phi , x , y ) = y ( \mathrm { p r o j } _ { w ^ { \perp } } x ) \mathbf { 1 } \{ | w \cdot x + \hat { \tau } \mathrm { c o s } ( \phi ) | \leq \sin ( \phi ) \}$ and $B _ { \phi } =$ $\{ z \in \mathbb { R } : | z + \hat { \tau } \mathrm { c o s } ( \phi ) | \leq \mathrm { s i n } ( \phi ) \}$ . Our GD algorithm outputs $f ( w , \phi , x , y ) : = \mathbb { E } _ { ( x , y ) \sim D } [ g ( w , \phi , x , y ) ] + z$ where z is the Gaussian noise (as in Claim 4.3).

Lemma 4.4 (Modification of Lemma C.12 from Diakonikolas et al. (2022)). Let $q = \operatorname* { P r } _ { x \sim \mu } [ { \boldsymbol { w } } \cdot { \boldsymbol { x } } \in B _ { \phi } ]$ Assuming that

$$
m _ { 0 } \ge \mathcal { O } \left( \frac { d q \log ( 1 / \beta _ { 0 } ) } { \eta ^ { 2 } } + \frac { \log ( d / \beta _ { 0 } ) } { q ^ { 2 } } + \frac { d \sqrt { \log ( 1 / \delta _ { \mathrm { D P } } ) \log ( 1 / \beta _ { 0 } ) } } { \eta \varepsilon _ { \mathrm { D P } } } \right) ,
$$

we have with probability at least 1− ${ \bf \nabla } \cdot \beta _ { 0 }$ that for any unit vector u $\in \mathbb { R } ^ { d }$ , such that $\mathbb { E } _ { ( x , y ) \sim P } [ g ( w , \phi , x , y )$ $u ] \geq 0$ , the following hold:

$$
f ( w , \phi , x , y ) \cdot u \geq \frac { 1 } { 2 } \mathbb { E } _ { ( x , y ) \sim P } [ g ( w , \phi , x , y ) \cdot u ] - \mathcal { O } \left( \eta \right)
$$

and

$$
\begin{array} { r } { \left. f ( w , \phi , x , y ) \right. \leq 2 \lVert \mathbb { E } _ { ( x , y ) \sim P } [ g ( w , \phi , x , y ) ] \rVert + \mathcal { O } \left( \eta \right) . } \end{array}
$$

Proof. The proof remains the same, except for accounting for the error due to Gaussian noise. This lemma is applied on independent dataset $D _ { t }$ in each iteration t. We use the notation D in this case as a shorthand for $D _ { t }$ , since we are omitting the subscript everywhere. For the first part, Diakonikolas et al. (2022) show the first line of the following.

$$
\begin{array} { r l } & { u \cdot \mathbb { E } _ { ( x , y ) \sim D } [ g ( w , \phi , x , y ) ] \geq u \cdot \mathbb { E } _ { ( x , y ) \sim P } [ g ( w , \phi , x , y ) ] \frac { \operatorname* { P r } _ { x \sim D _ { x } } [ w \cdot x \in B _ { \phi } ] } { \operatorname* { P r } _ { x \sim \mu } [ w \cdot x \in B _ { \phi } ] } - { \mathcal O } \left( \eta \right) \operatorname* { P r } _ { x \sim \mu } [ w \cdot x \in B _ { \phi } ] } \\ & { \qquad \implies u \cdot f ( w , \phi , x , y ) \geq u \cdot \mathbb { E } _ { ( x , y ) \sim P } [ g ( w , \phi , x , y ) ] \frac { \operatorname* { P r } _ { x \sim D _ { x } } [ w \cdot x \in B _ { \phi } ] } { \operatorname* { P r } _ { x \sim \mu } [ w \cdot x \in B _ { \phi } ] } - { \mathcal O } \left( \eta \right) \operatorname* { P r } _ { x \sim \mu } [ w \cdot x \in B _ { \phi } ] - \| z \| } \end{array}
$$

Our private sample complexity allows $\| z \| \leq \mathcal { O } \left( \eta \right)$ with probability at least $1 - \beta _ { 0 } / 2$ (via concentration of high-dimensional Gaussians). Therefore, this direction of the proof follows by following the remaining steps of the proof from Diakonikolas et al. (2022).

We justify the norm bound separately. Let $g _ { \mathrm { p o p } } : = \mathbb { E } [ g ( w , \phi , X , Y ) ]$ and $\widehat { g } : = \mathbb { E } _ { ( x , y ) \sim D } [ g ( w , \phi , x , y ) ]$ so that $f = { \widehat { g } } + z$ . Also, define the population and empirical masses of the band $B _ { \phi }$ by $q : =$ $\operatorname* { P r } _ { X \sim \mu } [ w \cdot X \in B _ { \phi } ]$ and $\textstyle { \widehat { q } } : = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ w \cdot x _ { i } \in B _ { \phi } \}$ , respectively. $\operatorname { I f } { \widehat { g } } = 0$ , the claim is immediate. Otherwise, define $u : = { \widehat { g } } / | { \widehat { g } } |$ . On the same concentration event on $z _ { \tau }$ used above, and the argument in Diakonikolas et al. (2022), writing $\widehat { q } / q \in [ 1 / 2 , 2 ]$ , their argument gives, for every unit vector $u ,$

$$
u \cdot \widehat { g } \leq \frac { \widehat { q } } { q } u \cdot g _ { \mathrm { p o p } } + \mathcal { O } \left( \eta \right) .
$$

If $\widehat g = 0$ , the desired bound is immediate. Otherwise, take $u = { \widehat { g } } / \| { \widehat { g } } \|$ . If $u \cdot g _ { \mathrm { p o p } } \geq 0$ , then

$$
\lVert \boldsymbol { \widehat { g } } \rVert \leq 2 u \cdot g _ { \mathrm { p o p } } + \mathcal { O } \left( \eta \right) \leq 2 \lVert g _ { \mathrm { p o p } } \rVert + \mathcal { O } \left( \eta \right) .
$$

If $u \cdot g _ { \mathrm { p o p } } < 0$ , then, since $\widehat { q } / q \geq 1 / 2$

$$
\frac { \widehat { q } } { q } u \cdot g _ { \mathrm { p o p } } \leq \frac { 1 } { 2 } u \cdot g _ { \mathrm { p o p } } < 0 ,
$$

and hence $\| \widehat { g } \| \leq \mathcal { O } \left( \eta \right)$ . Thus, in either case $\| \widehat { g } \| \leq 2 \| g _ { \mathrm { p o p } } \| + \mathcal { O } \left( \eta \right)$ . The bound $\| z \| \leq \mathcal { O } \left( \eta \right)$ implies

$$
\begin{array} { r } { \left. f \right. \leq \left. \widehat { g } \right. + \left. z \right. \leq 2 \lVert g _ { \mathrm { p o p } } \rVert + \mathcal { O } \left( \eta \right) , } \end{array}
$$

as claimed.

Now, writing $\theta = \theta ( w , w ^ { \star } )$ , we get $v \cdot \mathbb { E } _ { ( x , y ) \sim P } [ g ( w , \phi , x , y ) ] \ge \sin ( \theta ) e ^ { - t ^ { 2 } / 2 } > 0$ using Lemma C.3 from Diakonikolas et al. (2022). They then condition on the event $A _ { k }$ on which Lemma C.12 holds, and apply the directional result of Lemma C.12 to v. Hence, the only direction needed in their argument satisfies the positivity condition $( \mathbb { E } _ { ( x , y ) \sim P } [ g ( w , \phi , x , y ) \cdot u ] \ge 0 )$ required by our restricted statement. This lower bound on $v \cdot \mathbb { E } _ { ( x , y ) \sim P } [ g ( w , \phi , x , y ) ]$ is then converted into a lower bound on $w ^ { \star } \cdot \mathbb { E } _ { ( x , y ) \sim P } [ g ( w , \phi , x , y ) ]$ , and, together with the norm bound from Lemma C.12, is used to verify the hypotheses of Corollary C.8 and get contraction of the angle between the iterate and $w ^ { * }$

Next, we show that the DP initialization $w _ { 0 }$ in our algorithm satisfies the constraints that are needed in the proof of Theorem 1.2 from (Diakonikolas et al., 2022). They require that the dot product of the true weight $w ^ { * }$ with our initialization be strictly more than 0.

Claim 4.5. Suppose $\frac { e ^ { - \hat { \tau } ^ { 2 } / 2 } } { | \hat { \tau } | } \geq c \eta$ . Then, with probability at least $1 - \mathcal { O } \left( \beta \right)$ , if

$$
m _ { 0 } \ge \widetilde { \mathcal { O } } \left( \frac { d } { \eta ^ { 2 } } + \frac { \sqrt { d \log ( 1 / \delta _ { \mathrm { D P } } ) } } { \varepsilon _ { \mathrm { D P } } \eta } \right) ,
$$

the private initializer satisfies $w ^ { \star } \cdot \widehat { w } _ { 0 } > 0$

We only invoke Claim 4.5 in the above non-trivial regime. In the complementary case $\frac { e ^ { - \hat { \tau } ^ { 2 } / 2 } } { | \hat { \tau } | } < C \eta$ the argument of Diakonikolas et al. (2022) already implies that using the threshold $\hat { \tau }$ with any unit normal vector yields classification error $\mathcal { O } \left( \eta \right)$ . Hence, no positive-correlation initialization guarantee is needed in this case. Finally, we bound the error on our noisy empirical threshold.

Claim 4.6 (Modification of Claim C.9 from (Diakonikolas et al., 2022)). Suppose $\eta , \beta _ { 0 } \in ( 0 , 1 / 2 )$ ， and $w ^ { * }$ is a unit vector and τ is a threshold, such that $\operatorname* { P r } _ { ( x , y ) \sim P } [ \mathrm { s i g n } ( w ^ { * } \cdot x + \tau ) \neq y ] \leq \eta$ . Suppose $\hat { \tau } = \Phi ^ { - 1 } \left( \mathsf { c l i p } [ ( 1 + \mathbb { E } _ { ( x , y ) \sim D } [ y ] + z _ { \tau } ) / 2 ] _ { \kappa } ^ { 1 - \kappa } \right)$ for $\kappa = \mathcal { O } \left( \eta \right)$ is the DP estimate of the threshold $\tau$ output by Algorithm 3. If

$$
m _ { 0 } \geq \mathcal { O } \left( \log ( 1 / \beta _ { 0 } ) / \eta ^ { 2 } + \log ( 1 / \beta _ { 0 } ) / \varepsilon _ { \mathrm { D P } } \eta \right) ,
$$

then with probability at least $1 - \beta _ { 0 }$ 2

$$
\operatorname* { P r } _ { ( x , y ) \sim P } [ \mathrm { s i g n } ( w ^ { \ast } \cdot x + \hat { \tau } ) \neq y ] \leq \mathcal { O } \left( \eta \right) .
$$

Proof. Diakonikolas et al. (2022) show that satisfying $\left| \mathbb { E } _ { ( x , y ) \sim D } [ y ] - \mathbb { E } _ { ( x , y ) \sim P } [ y ] \right| \leq O \left( \eta \right)$ in the proof of Claim C.9 is enough, which is what we need, as well. Thus, $\left| \mathbb { E } _ { ( x , y ) \sim D } [ y ] ^ { \cdot } + z _ { \tau } - \mathbb { E } _ { ( x , y ) \sim P } [ y ] \right| \leq O \left( \eta \right)$ sufices for us, and the rest of the argument follows. We additionally clip the argument of $\Phi ^ { - 1 }$ to $[ \kappa , 1 - \kappa ]$ , where $\kappa = C \eta$ for a suficiently small universal constant $C > 0$

Let $\widetilde { p } = ( 1 + \mathbb { E } _ { ( x , y ) \sim \widehat { D } } [ y ] + z _ { \tau } ) / 2$ and $\widehat { p } = \mathsf { c l i p } ( \widetilde { p } , [ \kappa , 1 - \kappa ] )$ . Since $z _ { \tau }$ is sampled from Laplace distribution with parameter $2 / m _ { 0 } \varepsilon _ { \mathrm { D P } }$ , by the concentration of Laplace distribution, $| z _ { \tau } | \leq 2 \log ( 1 / \beta _ { 0 } ) / m _ { 0 } \varepsilon _ { \mathrm { D P } }$ with probability at least $1 - \beta _ { 0 }$ . For this to be upper bounded by $\mathcal { O } \left( \eta \right)$ , having $m _ { 0 } \geq \mathcal { O } \left( \log ( 1 / \beta _ { 0 } ) / \varepsilon _ { \mathrm { D P } } \eta \right)$ sufices. For any $p \in [ 0 , 1 ] , | \widehat { p } - p | \leq | \widetilde { p } - p | + \kappa$ . Hence, the clipping introduces at most an additional $\mathcal { O } \left( \eta \right)$ error. In particular, taking $p = ( 1 + \mathbb { E } _ { ( x , y ) \sim P } [ y ] ) / 2$ , the above concentration guarantee implies $\left| \widehat { p } - p \right| \leq \mathcal { O } \left( \eta \right)$ . Since $\widehat { \tau } = \Phi ^ { - 1 } ( \widehat { p } )$ , the remainder of the proof of Claim C.9 follows as before.

## 4.3 Putting It All Together

Proof of Proposition $4 . 2 .$ As mentioned earlier, our modifications are needed only for dataset-level results. Claims C.2 and C.9, along with Lemma C.12, are the only ones that needed modifications. Note that the technical results in Appendices B and C from Diakonikolas et al. (2022) still hold when we are using independent samples for threshold estimation, initialization, and each iteration of gradient descent, so they transfer to our private case, as well. We have shown via Claims 4.3 and 4.6, and Lemma 4.4, respectively, that at minimal cost in sample complexity, those results hold. Additionally, they prove that their initial weight vector $w _ { 0 }$ is not too far in terms of angle from $w ^ { * }$ We prove in Claim 4.5 that the same holds for our noisy initialization $\hat { w } _ { 0 }$ . With these necessary modifications, our proof of accuracy is complete. □

The grid in Algorithm 4 starts at $\eta _ { 1 } = C _ { \eta } \alpha$ , where $C _ { \eta } > 1$ is a universal constant, and uses $\eta _ { k } = 2 ^ { k - 1 } \eta _ { 1 }$ for $k \in [ K ]$ . The dataset is partitioned into a validation set $D _ { v a l }$ and K subsets $D ^ { 1 } , \ldots , D ^ { K }$ . For each $k \in [ K ]$ , the subroutine is run on $D ^ { k }$ with parameter $\eta _ { k }$ to obtain $h _ { k }$

Proof of Theorem $4 . 1 .$ Proposition F.1 provides the privacy guarantees. Now, we complete the proof of accuracy. First, note that the diference between the final error guarantee required to be in $\mathcal O \left( \alpha \right)$ , but in Algorithm 4, we work with $\eta _ { k } \mathrm { ^ { * } s }$ . The idea is that $\eta _ { k } \geq \alpha$ for all $k \in \left\lceil K \right\rceil$ . Now, either $\mathrm { O P T } < \eta _ { 1 }$ , or there must be a $k ^ { * } \in [ K ] \setminus \{ 1 \}$ , such that $\eta _ { 1 } 2 ^ { k ^ { * } - 2 } \leq \mathrm { O P T } \leq \eta _ { 1 } 2 ^ { k ^ { * } - 1 }$ . We claim that in either case, there exists at least one $k \in [ K ]$ , for which $h _ { k }$ has error $\mathcal { O } \left( \mathrm { O P T } + \alpha \right)$

When $\mathrm { O P T } < \eta _ { 1 }$ , then Algorithm 3 will return $h ,$ such that the error of h is at most $\mathcal { O } \left( \eta _ { 1 } \right) = \mathcal { O } \left( \alpha \right)$ with probability at least $1 - \beta ^ { \prime }$ , where $\beta ^ { \prime } = \beta / K$ (Proposition 4.2). This satisfies the semi-agnostic learning guarantee even if OPT is much smaller than η because $\alpha = \Theta ( \eta _ { 1 } )$

On the other hand, suppose there exists $k ^ { * } \in [ K ] \setminus \{ 1 \}$ , such that $\eta _ { 1 } 2 ^ { k ^ { * } - 2 } \leq \mathrm { O P T } \leq \eta _ { 1 } 2 ^ { k ^ { * } - 1 }$ Then with probability at least $1 - ( K - 1 ) \beta ^ { \prime }$ , for all $k \geq k ^ { * }$ , Algorithm 3 will return hypotheses $h _ { k } .$

respectively, such that the total error is $\mathcal { O } \left( \eta _ { k } \right)$ . Thus for $k ^ { \ast } , \eta _ { k ^ { \ast } } = \eta _ { 1 } 2 ^ { k ^ { \ast } - 1 } \leq 2 \mathrm { O P T }$ . This means that $h _ { k ^ { * } }$ should satisfy the semi-agnostic error guarantee $\mathcal { O } \left( \mathrm { O P T } \right)$ with no additional additive error.

Combining the two together, with probability at least $1 - \mathcal { O } \left( \beta \right)$ , there must exist $k \in [ K ]$ , such that the semi-agnostic error of $h _ { k }$ is at most max $\left\{ \mathcal O \left( \alpha \right) , \mathcal O \left( \mathrm { O P T } \right) \right\}$ . Now, we just need to show that the final hypothesis returned by Algorithm 4 is accurate. For that, we know that with probability at least $1 - \mathcal { O } \left( \beta \right)$ , for all $k \in [ K ]$ , the population error $\mathrm { e r r } _ { P } ( h _ { k } )$ must be close to the empirical error $\widehat { \mathrm { e r r } } _ { D _ { v a l } } ( h _ { k } )$ on $D _ { v a l }$ using Hoefding’s inequality; because of our sample complexity,

$$
| \mathrm { e r r } _ { P } ( h _ { k } ) - \widehat { \mathrm { e r r } } _ { D _ { v a l } } ( h _ { k } ) | \leq \mathcal { O } \left( \sqrt { \frac { \log ( K / \beta ) } { | D _ { v a l } | } } \right) \leq \mathcal { O } \left( \alpha \right) .
$$

With probability at least $1 - \mathcal { O } \left( \beta \right)$ , the exponential mechanism returns an $\bar { h } .$ , whose empirical error is at most $\mathcal O \left( \alpha \right)$ in addition to the empirical optimal among all $h _ { k } \mathrm { { ' s } }$ . From these two additional error sources, we get that h<sup>¯</sup> has a population error of at most max $\left\{ \mathcal O \left( \alpha \right) , \mathcal O \left( \mathrm { O P T } \right) \right\} + \mathcal O \left( \alpha \right)$

Finally, we bound the running time. Each run of Algorithm 3 uses every sample in its data a constant number of times (for clipping, the estimates of τˆ and $\hat { w } _ { 0 }$ , or one gradient step), at $\mathcal O \left( d \right)$ cost per use. Since the runs use disjoint data, all $K = \mathcal { O } \left( \log ( 1 / \alpha ) \right)$ runs together take $\mathcal { O } \left( n d \right)$ time. Scoring the K candidates on $D _ { v a l }$ takes $\mathcal { O } \left( n d \log ( 1 / \alpha ) \right)$ time. Since $d \leq n$ , the total running time is $\mathcal { O } \left( n d \log ( 1 / \alpha ) \right) = \mathrm { p o l y } ( n )$ □

## 5 Conclusion and Open Problems

We provide the first computationally eficient DP agnostic learners for halfspaces, where privacy comes for free in terms of sample complexity, i.e., the sample complexity overhead due to privacy matches that of the non-private learners. Specifically, we study three settings: improper agnostic, proper agnostic, and semi-agnostic. Our techniques are based on those of the non-private work, albeit with additional steps and analyses to incorporate clipping and noise injection to provide DP guarantees.

One avenue for improvement is in the improper agnostic setting, where we have to explicitly compute the orthonormal basis for each family of distributions when the marginals are known, unlike Kalai et al. (2008), who work with the standard polynomial basis for all families of distributions. The only parameter they need to know for their kernel is the degree of the polynomial. An interesting future direction is to find more versatile polynomial bases that admit bounded coeficients for a large family of distributions, or alternatively to learn such a basis privately from the data, removing the need to know the marginal in advance. Another direction is to design eficient DP algorithms for agnostic learning of concept classes beyond halfspaces. Finally, one could explore lower bounds in DP agnostic learning to determine the necessary sample complexity overhead for these problems.

## Acknowledgements and AI Disclosure

The authors acknowledge the Novo Nordisk Foundation for support via the Startup grant (NNF24OC0 087820); AS additionally acknowledges support from VILLUM FONDEN via the Young Investigator program (VIL72069).

The main ideas underlying this work were developed by the authors. AI tools were used to help sharpen the parameter dependence of technical lemmas. The authors take full responsibility for the correctness of the results and the content of the final manuscript.

## References

Noga Alon, Roi Livni, Maryanthe Malliaris, and Shay Moran. Private PAC learning implies finite Littlestone dimension. In ACM Symposium on Theory of Computing (STOC), 2019.

Noga Alon, Amos Beimel, Shay Moran, and Uri Stemmer. Closure properties for private classification and online prediction. In Conference on Learning Theory (COLT), 2020.

George E Andrews, Richard Askey, and Ranjan Roy. Special functions. 1999.

Pranjal Awasthi, Maria Florina Balcan, and Philip M Long. The power of localization for eficiently learning linear separators with noise. Journal of the ACM (JACM), 2017.

Raef Bassily, Mehryar Mohri, and Ananda Theertha Suresh. Diferentially private learning with margin guarantees. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Amos Beimel, Shay Moran, Kobbi Nissim, and Uri Stemmer. Private center points and learning of halfspaces. In Conference on Learning Theory (COLT), 2019.

Mark Bun, Kobbi Nissim, Uri Stemmer, and Salil Vadhan. Diferentially private release and learning of threshold functions. In IEEE Symposium on Foundations of Computer Science (FOCS), 2015.

Mark Bun, Roi Livni, and Shay Moran. An equivalence between private classification and online prediction. In IEEE Symposium on Foundations of Computer Science (FOCS), 2020.

Amit Daniely. A PTAS for agnostically learning halfspaces. In Conference on Learning Theory (COLT), 2015.

Amit Daniely. Complexity theoretic limitations on learning halfspaces. In ACM Symposium on Theory of Computing (STOC), 2016.

Ilias Diakonikolas, Daniel M Kane, and Jelani Nelson. Bounded independence fools degree-2 threshold functions. In IEEE Symposium on Foundations of Computer Science (FOCS), 2010. URL https://arxiv.org/abs/0911.3389.

Ilias Diakonikolas, Daniel M Kane, and Alistair Stewart. Learning geometric concepts with nasty noise. In ACM Symposium on Theory of Computing (STOC), 2018.

Ilias Diakonikolas, Daniel Kane, and Nikos Zarifis. Near-optimal SQ lower bounds for agnostically learning halfspaces and ReLUs under Gaussian marginals. In Advances in Neural Information Processing Systems (NeurIPS), 2020a.

Ilias Diakonikolas, Vasilis Kontonis, Christos Tzamos, and Nikos Zarifis. Non-convex SGD learns halfspaces with adversarial label noise. In Advances in Neural Information Processing Systems (NeurIPS), 2020b.

Ilias Diakonikolas, Daniel M. Kane, Vasilis Kontonis, Christos Tzamos, and Nikos Zarifis. Agnostic proper learning of halfspaces under Gaussian marginals. In Conference on Learning Theory (COLT), 2021a. URL https://proceedings.mlr.press/v134/diakonikolas21b.html.

Ilias Diakonikolas, Daniel M. Kane, Thanasis Pittas, and Nikos Zarifis. The optimality of polynomial regression for agnostic learning under Gaussian marginals in the SQ model. In Conference on Learning Theory (COLT), 2021b.

Ilias Diakonikolas, Vasilis Kontonis, Christos Tzamos, and Nikos Zarifis. Learning general halfspaces with adversarial label noise via online gradient descent. In International Conference on Machine Learning (ICML), 2022. URL https://proceedings.mlr.press/v162/diakonikolas22b.html.

Ilias Diakonikolas, Daniel Kane, and Lisheng Ren. Near-optimal cryptographic hardness of agnostically learning halfspaces and ReLU regression under Gaussian marginals. In International Conference on Machine Learning (ICML), 2023.

Charles F. Dunkl and Yuan Xu. Orthogonal Polynomials of Several Variables. 2014.

Cynthia Dwork and Aaron Roth. The algorithmic foundations of diferential privacy. Foundations and Trends in Theoretical Computer Science, 2014.

Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography Conference (TCC), 2006.

Vitaly Feldman, Parikshit Gopalan, Subhash Khot, and Ashok Kumar Ponnuswami. New results for learning noisy parities and halfspaces. In IEEE Symposium on Foundations of Computer Science (FOCS), 2006.

Spencer Frei, Yuan Cao, and Quanquan Gu. Agnostic learning of halfspaces with gradient descent via soft margins. In International Conference on Machine Learning (ICML), 2021.

G Freud, A Giroux, and QI Rahman. Sur l’approximation polynomiale avec poids exp(−|x|). Canadian Journal of Mathematics, 1978.

Surbhi Goel, Aravind Gollakota, and Adam Klivans. Statistical-query lower bounds via functional gradients. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

Venkatesan Guruswami and Prasad Raghavendra. Hardness of learning halfspaces with noise. SIAM Journal on Computing, 2009.

Wassily Hoefding. Probability inequalities for sums of bounded random variables. Journal of the American statistical association, 1963.

Ziwei Ji, Kwangjun Ahn, Pranjal Awasthi, Satyen Kale, and Stefani Karp. Agnostic learnability of halfspaces via logistic loss. In International Conference on Machine Learning (ICML), 2022.

Adam Tauman Kalai, Adam R. Klivans, Yishay Mansour, and Rocco A. Servedio. Agnostically learning halfspaces. SIAM Journal on Computing, 2008. URL https://doi.org/10.1137/ 060649057.

Gautam Kamath, Jerry Li, Vikrant Singhal, and Jonathan Ullman. Privately learning highdimensional distributions. In Conference on Learning Theory (COLT), 2019.

Gautam Kamath, Argyris Mouzakis, and Vikrant Singhal. New lower bounds for private estimation and a generalized fingerprinting lemma. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Haim Kaplan, Yishay Mansour, Uri Stemmer, and Eliad Tsfadia. Private learning of halfspaces: Simplifying the construction and reducing the sample complexity. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

Vishesh Karwa and Salil Vadhan. Finite sample diferentially private confidence intervals. In Conference on Innovations in Theoretical Computer Science (ITCS), 2018.

Shiva Prasad Kasiviswanathan, Homin K. Lee, Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith. What can we learn privately? In IEEE Symposium on Foundations of Computer Science (FOCS), 2008.

Michael Kearns. Eficient noise-tolerant learning from statistical queries. Journal of the ACM (JACM), 1998.

Janardhan Kulkarni, Yin Tat Lee, and Daogao Liu. Private non-smooth ERM and SCO in subquadratic steps. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Bo Li, Wei Wang, and Peng Ye. Private realizable-to-agnostic transformation with near-optimal sample complexity. In Conference on Learning Theory (COLT), 2025.

Nathan Linial, Yishay Mansour, and Noam Nisan. Constant depth circuits, Fourier transform, and learnability. Journal of the ACM (JACM), 1993.

Nick Littlestone. Learning quickly when irrelevant attributes abound: A new linear-threshold algorithm. Machine learning, 1988.

Doron S Lubinsky. A survey of weighted approximation for exponential weights. arXiv preprint math/0701099, 2007.

Frank McSherry and Kunal Talwar. Mechanism design via diferential privacy. In IEEE Symposium on Foundations of Computer Science (FOCS), 2007. URL https://doi.org/10.1109/FOCS.2007.66.

HN Mhaskar and EB Saf. Extremal problems for polynomials with exponential weights. Transactions of the American Mathematical Society, 1984.

Huy Lê Nguy<sup>˜</sup>ên, Jonathan Ullman, and Lydia Zakynthinou. Eficient private algorithms for learning large-margin halfspaces. In International Conference on Algorithmic Learning Theory (ALT). PMLR, 2020.

Ryan O’Donnell. Analysis of Boolean Functions. 2014.

Lucas Pesenti, Lucas Slot, and Manuel Wiedmer. Agnostic learning in (almost) optimal time via Gaussian surface area. arXiv preprint arXiv:2603.06027, 2026.

Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: From theory to algorithms. 2014.

Shai Shalev-Shwartz, Ohad Shamir, Nathan Srebro, and Karthik Sridharan. Learnability, stability and uniform convergence. The Journal of Machine Learning Research, 2010.

Stefan Tiegel. Hardness of agnostically learning halfspaces from worst-case lattice problems. In Conference on Learning Theory (COLT), 2023. URL https://proceedings.mlr.press/v195/ tiegel23a.html.

## A Other Related Work

Agnostic learning under structured marginals. Without distributional assumptions, agnostically learning halfspaces is computationally hard, even for weak learning (Guruswami and Raghavendra, 2009; Feldman et al., 2006; Daniely, 2016; Tiegel, 2023). Building on the low-degree Fourier approach of Linial et al. (1993), Kalai et al. (2008) use polynomial $L _ { 1 }$ regression to obtain improper agnostic learners for halfspaces with error OPT +α under structured marginals, including Gaussian and log-concave distributions. The complexity of this approach is governed by the degree of polynomials needed to approximate halfspaces in $L _ { 1 }$ . Under Gaussian marginals, Diakonikolas et al. (2010) improved the runtime of Kalai et al. (2008) from $d ^ { O ( \alpha ^ { - 4 } ) } \ \mathrm { t o } \ d ^ { O ( \breve { \alpha } ^ { - 2 } ) }$ by establishing sharper polynomial approximation bounds. This is essentially optimal: statistical-query algorithms require time $d ^ { \Omega ( \alpha ^ { - 2 } ) }$ (Goel et al., 2020; Diakonikolas et al., 2020a, 2021b), and a nearly matching lower bound holds for all polynomial-time algorithms under the subexponential hardness of Learning with Errors (Diakonikolas et al., 2023). However, this approach is improper, as the resulting classifier need not be a halfspace. Diakonikolas et al. (2021a) addressed this limitation by using polynomial regression to identify a subspace containing the normal vector of a near-optimal halfspace; this yields a proper learner with error $\mathrm { O P T } + \alpha$ and runtime $d ^ { \mathrm { p o l y } ( 1 / \alpha ) }$

Semi-agnostic learning. Motivated by these computational barriers, a related line of work improves computational eficiency by relaxing the agnostic error guarantee. For homogeneous halfspaces, Awasthi et al. (2017) developed a localization-based algorithm achieving $O ( \mathrm { O P T } ) + \alpha$ error under isotropic log-concave marginals. Combining localization with polynomial regression, Daniely (2015) obtains an improper approximation scheme under the uniform spherical marginal, achieving error $\left( 1 + \gamma \right) \mathrm { O P T } + \alpha$ in time polynomial in d and $1 / \alpha$ for every fixed $\gamma > 0 ;$ Diakonikolas et al. (2021a) give a proper analogue under Gaussian marginals. Diakonikolas et al. (2020b) showed that non-convex SGD achieves $O ( \mathrm { O P T } ) + \alpha$ error for homogeneous halfspaces under a broad family of marginals, including isotropic log-concave distributions. For general halfspaces under Gaussian marginals, Diakonikolas et al. (2018) obtained constant-factor guarantees, even in a stronger contamination model allowing corruption of both examples and labels, and Diakonikolas et al. (2022) developed the sample-eficient online gradient descent algorithm on which our private semi-agnostic learner builds. Complementary work studies convex surrogates for homogeneous halfspaces. Under well-behaved marginals, including isotropic log-concave distributions, Frei et al. (2021) obtain $\widetilde { O } ( \sqrt { \mathrm { O P T } } )$ + α error using gradient methods on convex losses, while Ji et al. (2022) show that logistic regression alone cannot improve on this and obtain $\widetilde { O } ( \mathrm { O P T } ) + \alpha$ by adding a hinge-loss refinement step.

Private learning. Distribution-free private learning of halfspaces over infinite domains is impossible: already thresholds over $\mathbb { R }$ are not privately learnable (Bun et al., 2015), and private learnability is characterized by finite Littlestone dimension (Littlestone, 1988; Alon et al., 2019; Bun et al., 2020). Over finite domains $X \subset \mathbb { R } ^ { d }$ , Beimel et al. (2019) (improved by Kaplan et al. (2020)) give private agnostic learners whose sample complexity grows with log<sup>∗</sup> |X|. Generic private realizable-to-agnostic transformations (Alon et al., 2020; Li et al., 2025) are statistically eficient but not computationally eficient in general. Eficient private PAC learners exist under margin assumptions (Nguy<sup>˜</sup>ên et al., 2020; Bassily et al., 2022), with guarantees stated for margin-based losses rather than OPT for the 0-1 loss.

## B Auxiliary Results

## B.1 Concentration Inequalities

Lemma B.1 (Vector Bernstein Inequality). Let $U _ { 1 } , \dots , U _ { n }$ be independent, mean-zero random vectors in $\mathbb { R } ^ { d }$ . Suppose that $\| U _ { i } \| _ { 2 } \le B$ almost surely for all $i \in \{ 1 , \ldots , n \}$ , and let $\begin{array} { r } { \sum _ { i = 1 } ^ { n } \mathbb { E } \| U _ { i } \| _ { 2 } ^ { 2 } \leq v } \end{array}$ Then, for every $\beta \in ( 0 , 1 )$ , with probability at least $1 - \beta$

$$
\left\| { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } U _ { i } \right\| _ { 2 } \leq { \frac { \sqrt { 2 v \log ( 2 / \beta ) } } { n } } + { \frac { 2 B \log ( 2 / \beta ) } { 3 n } } .
$$

## B.2 Diferential Privacy Tools

Definition B.2 (Diferential privacy; Dwork et al., 2006). Let $\varepsilon _ { \mathrm { D P } } > 0$ and $\delta _ { \mathrm { D P } } ~ \in ~ [ 0 , 1 )$ . A randomized algorithm $\mathcal { A } : \mathcal { Z } ^ { n } \to \mathcal { V }$ is $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) - d i f f$ erentially private if, for every pair of datasets $S , S ^ { \prime } \in \mathcal { Z } ^ { n }$ difering in a single example and every measurable set $E \subseteq \mathcal { V }$ 2

$$
\mathrm { P r } [  { \boldsymbol { A } } (  { \boldsymbol { S } } ) \in E ] \leq e ^ { \varepsilon _ { \mathrm { D P } } } \ \mathrm { P r } [  { \boldsymbol { A } } (  { \boldsymbol { S } } ^ { \prime } ) \in E ] + \delta _ { \mathrm { D P } } .
$$

Lemma B.3 (Post-Processing of DP; Dwork and Roth, 2014). If $M : \mathcal { X } ^ { n }  \mathcal { Y }$ is $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ and $f : \mathcal { y }  \mathcal { Z }$ is any randomized function, then the algorithm $f \circ M$ is $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) { \cdot } \mathrm { D P } .$

Lemma B.4 (Private convex ERM; Kulkarni et al., 2021, Theorem 1.1). Let $\Omega \subseteq \mathbb { R } ^ { D }$ be a public, nonempty, compact convex set of Euclidean diameter R, and let $\ell ( \cdot ; z ) : \mathbb { R } ^ { D } \to$ R be a public convex T-Lipschitz loss for every $z \in { \mathcal { Z } }$ . For $n \geq 2 , 0 < \varepsilon _ { \mathrm { D P } } \leq 1 / 2$ , and $0 < \delta _ { \mathrm { D P } } \leq 1 / 2$ , there exists an $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ algorithm under replacement adjacency such that, for every $S = ( Z _ { 1 } , \ldots , Z _ { n } ) \in \mathcal { Z } ^ { n }$ ， its output $\widehat { v } \in \Omega$ satisfies

$$
\mathbb { E } \bigg [ \widehat { L } _ { S } ( \widehat { v } ) \bigg ] - \operatorname* { m i n } _ { v \in \Omega } \widehat { L } _ { S } ( v ) \leq \mathcal { O } \left( \frac { T R \sqrt { D \log ( 1 / \delta _ { \mathrm { D P } } ) } } { n \varepsilon _ { \mathrm { D P } } } \right) ,
$$

where $\textstyle { \widehat { L } } _ { S } ( v ) : = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \ell ( v ; Z _ { i } )$ , and the expectation is over the algorithm’s randomness. The algorithm performs $\mathcal { O } \left( n ^ { 2 } \right)$ subgradient evaluations and Euclidean projections onto Ω.

Proof. Compared to Kulkarni et al. (2021, Theorem 1.1), we use $R$ as an upper bound on the initialization distance and assume convexity and T-Lipschitzness on all of $\mathbb { R } ^ { D }$ , rather than only on the neighborhood required in the original statement. Their complexity guarantee implies $\mathcal { O } \left( n ^ { 2 } \right)$ subgradient evaluations in the stated parameter range, with at most as many Euclidean projections, since each iteration uses one projection and at least one subgradient evaluation. □

Lemma B.5 (Validation via the exponential mechanism). Let $K , m \in \mathbb { N } ,$ , and let $h _ { 1 } , \ldots , h _ { K }$ be classifiers independent of $S _ { \mathrm { v a l } } \sim P ^ { m }$ . For $\varepsilon _ { \mathrm { D P } } > 0$ , select $\hat { j } \in [ K ]$ with probability proportional to $\mathrm { e x p } ( - \varepsilon _ { \mathrm { D P } } m \widehat { \mathrm { e r r } } _ { S _ { \mathrm { v a l } } } ( h _ { j } ) / 2 )$ . Conditional on the candidates, selection is ε -diferentially private under replacement. For $0 < \eta < 1$ , with probability at least $1 - \eta$

$$
\mathrm { e r r } _ { P } ( h _ { \widehat { j } } ) \leq \operatorname* { m i n } _ { j \in [ K ] } \mathrm { e r r } _ { P } ( h _ { j } ) + 2 \sqrt { \frac { \log ( 4 K / \eta ) } { 2 m } } + \frac { 2 \log ( 2 K / \eta ) } { \varepsilon _ { \mathrm { D P } } m } .
$$

Proof. Condition on the candidates. Hoefding’s inequality (Hoefding, 1963) and a union bound give the first slack term, accounting for the deviations of validation errors from population errors for both the selected classifier and the best candidate. The score $- m { \widehat { \mathrm { e r r } } } _ { S _ { \mathrm { v a l } } } ( h _ { j } )$ has sensitivity one, so the exponential mechanism ensures privacy and gives the second slack term, accounting for suboptimal selection based on empirical errors (McSherry and Talwar, 2007). Allocate failure probability $\eta / 2$ to each bound. □

## B.3 $L ^ { 1 }$ Regression

Fact B.6 (Random thresholding; Kalai et al., 2008, Proof of Theorem 5). Let $\tau \sim \mathrm { U n i f } [ - 1 , 1 ]$ . For every $z \in \mathbb { R }$ and $y \in \{ - 1 , 1 \}$ , we have $\begin{array} { r } { \operatorname* { P r } _ { \tau } \left[ \mathrm { s i g n } ( z - \tau ) \neq y \right] \leq \frac { 1 } { 2 } \left| z - y \right| } \end{array}$

Lemma B.7 (Uniform deviation for $L ^ { 1 }$ regression). Let $\varphi : \mathcal { X } \to \mathbb { R } ^ { D }$ be a fixed feature map such that $\begin{array} { r } { \mathbb { E } \| \varphi ( X ) \| _ { 2 } ^ { 2 } \leq D } \end{array}$ . Let $L _ { P } ( v ) = \mathbb { E } | \left. v , \varphi ( X ) \right. - Y |$ , and let $\widehat { L } _ { S } ( v )$ be its empirical counterpart on $S \sim P ^ { n }$ . Then, for every $R > 0$

$$
\mathbb { E } _ { S \sim P ^ { n } } \operatorname* { s u p } _ { \| v \| _ { 2 } \leq R } \Big ( L _ { P } ( v ) - \widehat { L } _ { S } ( v ) \Big ) \leq 2 R \sqrt { \frac { D } { n } } .
$$

Proof. Let $\sigma _ { 1 } , \ldots , \sigma _ { n }$ be i.i.d. Rademacher random variables, independent of S. Symmetrization (Shalev-Shwartz and Ben-David, 2014, Lemma 26.2) and contraction for the 1-Lipschitz maps $t \mapsto | t - Y _ { i } |$ (Shalev-Shwartz and Ben-David, 2014, Lemma 26.9) give

$$
\mathbb { E } _ { S } \operatorname* { s u p } _ { \| v \| _ { 2 } \leq R } \Big ( L _ { P } ( v ) - \widehat { L } _ { S } ( v ) \Big ) \leq \frac { 2 R } { n } \mathbb { E } _ { S , \sigma } \left\| \sum _ { i = 1 } ^ { n } \sigma _ { i } \varphi ( X _ { i } ) \right\| _ { 2 } .
$$

By Jensen’s inequality, and since the cross terms vanish in expectation, the right-hand side is at most $\begin{array} { r } { \frac { 2 R } { n } \big ( \sum _ { i = 1 } ^ { n } \mathbb { E } \left\| \varphi ( X _ { i } ) \right\| _ { 2 } ^ { 2 } \big ) ^ { 1 / 2 } \leq 2 R \sqrt { D / n } } \end{array}$ □

## B.4 Gaussian Marginals

For a probability measure µ on $\mathbb { R } ^ { d }$ and $r \in [ 1 , \infty )$ , we write $\begin{array} { r } { \| f \| _ { L ^ { r } ( \mu ) } : = \left( \int | f ( x ) | ^ { r } d \mu ( x ) \right) ^ { 1 / r } } \end{array}$ and $\| f \| _ { L ^ { \infty } ( \mu ) } : = \operatorname* { i n f } \big \{ M \geq 0 : \mu ( \{ x : | f ( x ) | > M \} ) = 0 \big \}$ . The inner product on $L ^ { 2 } ( \mu )$ is $\langle f , g \rangle _ { L ^ { 2 } ( \mu ) } : =$ $\begin{array} { r } { \int f ( x ) g ( x ) d \mu ( x ) } \end{array}$

Lemma B.8 (Hermite basis; O’Donnell, 2014, Proposition 11.33). Let $\{ h _ { m } \} _ { m \geq 0 }$ be the normalized univariate Hermite polynomials, defined by

$$
e ^ { t x - t ^ { 2 } / 2 } = \sum _ { m = 0 } ^ { \infty } h _ { m } ( x ) \frac { t ^ { m } } { \sqrt { m ! } } .
$$

For each multi-index ${ \kappa } = ( { \kappa } _ { 1 } , \ldots , { \kappa } _ { d } ) \in \mathbb { N } _ { 0 } ^ { d } .$ , define $\begin{array} { r } { h _ { \kappa } ( x ) = \prod _ { i = 1 } ^ { d } h _ { \kappa _ { j } } ( x _ { j } ) } \end{array}$ and $\begin{array} { r } { | \kappa | = \sum _ { j = 1 } ^ { d } \kappa _ { j } } \end{array}$ . Then $\{ h _ { \kappa } \} _ { \kappa \in  { \mathbb { N } } _ { 0 } ^ { d } }$ is a complete orthonormal basis of $L ^ { 2 } ( \mathcal { N } ( 0 , I _ { d } ) )$ . In particular, $\{ h _ { \kappa } : | \kappa | \le p \}$ is an orthonormal basis for the polynomials of total degree at most $p ,$ with dimension $\textstyle { \binom { d + p } { p } }$

Lemma B.9 (Mehler’s formula; Andrews et al., 1999, Eq. 6.1.13). Let $\{ h _ { \kappa } \} _ { \kappa \in  { \mathbb { N } } _ { 0 } ^ { d } }$ be the multivariate Hermite polynomials, orthonormal in $L ^ { 2 } ( \mathcal { N } ( 0 , I _ { d } ) )$ . For each multi-index $\kappa \in \mathbb { N } _ { 0 } ^ { d }$ , write $\begin{array} { r } { | \kappa | = \sum _ { j = 1 } ^ { d } \kappa _ { j } } \end{array}$ For every $x , y \in \mathbb { R } ^ { d }$ and $\rho \in ( - 1 , 1 )$

$$
\sum _ { \kappa \in \mathbb { N } _ { 0 } ^ { d } } \rho ^ { | \kappa | } h _ { \kappa } ( x ) h _ { \kappa } ( y ) = \frac { 1 } { ( 1 - \rho ^ { 2 } ) ^ { d / 2 } } \exp \left( \frac { 2 \rho \left. x , y \right. - \rho ^ { 2 } \left( \left\| x \right\| _ { 2 } ^ { 2 } + \left\| y \right\| _ { 2 } ^ { 2 } \right) } { 2 ( 1 - \rho ^ { 2 } ) } \right) .
$$

Lemma B.10 (Unit-norm polynomial approximation of halfspaces under Gaussian measure). Let $d \geq 1$ , and let $\{ h _ { \kappa } \} _ { \kappa \in  { \mathbb { N } } _ { 0 } ^ { d } }$ be the orthonormal Hermite basis of $L ^ { 2 } ( \mathcal { N } ( 0 , I _ { d } ) )$ . There exists a universal constant $C > 0$ such that for every $\alpha \in ( 0 , 1 )$ and every halfspace $c : \mathbb { R } ^ { d }  \{ - 1 , 1 \}$ , there is a polynomial $\begin{array} { r } { q ( x ) = \sum _ { | \kappa | \leq p } w _ { \kappa } h _ { \kappa } ( x ) } \end{array}$ of degree $p \leq C \alpha ^ { - 2 }$ with $\| w \| _ { 2 } \leq 1$ and $\| c - q \| _ { L ^ { 1 } ( \mathcal { N } ( 0 , I _ { d } ) ) } \leq \alpha$

Proof. Write $c ( x ) = f _ { \theta } ( \langle u , x \rangle )$ , where $f _ { \theta } ( z ) = \operatorname { s g n } ( z - \theta )$ and $\| u \| _ { 2 } = 1$ . Let $B : \mathbb { R } \to [ 0 , \infty )$ be the one-dimensional smoothing kernel of Diakonikolas et al. (2010, Section 4), and define $B _ { \lambda } ( t ) = \lambda B ( \lambda t )$

By Pesenti et al. (2026, Section 4.2), for every $\theta \in \mathbb { R }$ , there exist $\lambda > 0$ and an integer $p \leq C \alpha ^ { - 2 }$ 2 with C independent of $\theta ,$ such that $g _ { \theta } = B _ { \lambda } * f _ { \theta }$ satisfies

$$
\begin{array} { r } { \left. f _ { \theta } - \Pi _ { p } g _ { \theta } \right. _ { L ^ { 1 } ( \mathcal { N } ( 0 , 1 ) ) } \le \alpha , } \end{array}
$$

where $\Pi _ { p } ( \cdot )$ is the $L ^ { 2 } ( \mathcal { N } ( 0 , 1 ) )$ )-orthogonal projection onto polynomials of degree at most $p .$

Since $\begin{array} { r } { \int _ { \mathbb { R } } B _ { \lambda } ( t ) d t = 1 } \end{array}$ (Diakonikolas et al., 2010, Lemma 4.4), we have

$$
| g _ { \theta } ( z ) | \leq \int _ { \mathbb { R } } B _ { \lambda } ( t ) | f _ { \theta } ( z - t ) | d t = 1 \quad \forall z .
$$

Hence $\| \Pi _ { p } g _ { \theta } \| _ { L ^ { 2 } ( \mathcal { N } ( 0 , 1 ) ) } \leq \| g _ { \theta } \| _ { L ^ { 2 } ( \mathcal { N } ( 0 , 1 ) ) } \leq 1$

Set $q ( x ) = \Pi _ { p } g _ { \theta } ( \langle u , x \rangle )$ . Its degree is at most $p ,$ so $\begin{array} { r } { q \ = \ \sum _ { | \kappa | \leq p } w _ { \kappa } h _ { \kappa } } \end{array}$ for some coeficient vector w (Lemma B.8). Since $\| u \| _ { 2 } = 1$ , we have $\langle u , X \rangle \sim \mathcal { N } ( 0 , 1 )$ for $X \sim { \mathcal { N } } ( 0 , I _ { d } )$ . Thus, the approximation error and $L ^ { 2 }$ norm of q under $\mathcal { N } ( 0 , I _ { d } )$ equal those of $\Pi _ { p } g _ { \theta }$ under $\mathcal { N } ( 0 , 1 )$ . Finally, orthonormality yields $\| w \| _ { 2 } = \| q \| _ { L ^ { 2 } ( \mathcal { N } ( 0 , I _ { d } ) ) } = \| \Pi _ { p } g _ { \theta } \| _ { L ^ { 2 } ( \mathcal { N } ( 0 , 1 ) ) } \leq 1$ □

Fact B.11 (Euclidean covers; Diakonikolas et al., 2021a, Fact 21). For every $k \geq 1$ and $\gamma \in ( 0 , 1 )$ ， the unit sphere in $\mathbb { R } ^ { k }$ has an explicit γ-cover in $\ell _ { 2 }$ of size at most $( C / \gamma ) ^ { k }$ , where $C > 0$ is a universal constant.

Fact B.12 (Gaussian halfspace discretization; Diakonikolas et al., 2021a, Fact 22). Let $X \sim N ( 0 , I _ { d } )$ 2 let $u , v \in \mathbb { R } ^ { d }$ be unit vectors, and let $t _ { 1 } , t _ { 2 } \in \mathbb { R }$ . Then $\mathbb { E } [ \left| h _ { u , t _ { 1 } } ( X ) - h _ { v , t _ { 1 } } ( X ) \right| ] = { \mathcal { O } } \left( \left\| u - v \right\| _ { 2 } \right)$ and $\mathbb { E } [ \left| h _ { u , t _ { 1 } } ( X ) - h _ { u , t _ { 2 } } ( X ) \right| ] = { \mathcal { O } } \left( \left| t _ { 1 } - t _ { 2 } \right| \right)$ . Moreover, if $| t | \geq C \sqrt { \log ( 1 / \gamma ) }$ , then $\mathbb { E } [ | h _ { v , t } ( X ) - \mathrm { s i g n } ( t ) | ] =$ $\mathcal { O } \left( \gamma \right)$

## B.5 Log-Concave Marginals

A probability measure is log-concave if its density is of the form $e ^ { - V }$ , where V is convex. Logconcavity is preserved under linear projections, and log-concave distributions have finite moments of all orders. We use the following quantitative density bound for standardized one-dimensional log-concave distributions.

Fact B.13 (Bound on the density of log-concave distributions; Kalai et al., 2008, Lemma 6). Let ν be a log-concave distribution on R with mean zero and variance one. Then its density g satisfies $g ( x ) \leq e \cdot e ^ { - | x | / 1 6 }$ for all $x \in \mathbb { R }$

The next two results concern weighted polynomial approximation on R with the exponential weight function. They are not specific to log-concave measures. We use them, together with Fact B.13, in the proof of Lemma D.3. Let $\| \cdot \| _ { L ^ { k } }$ denote the $L ^ { k }$ norm with respect to the Lebesgue measure, and let $W ( x ) : = e ^ { - | x | }$

Lemma B.14 (Weighted Nikolskii inequality; Lubinsky, 2007, §9). There is a universal constant C such that, for every $p \geq 1$ and every univariate polynomial $q$ of degree at most $p ,$

$$
\| q W \| _ { L ^ { \infty } } \leq C \log ( p + 1 ) \ \| q W \| _ { L ^ { 1 } } .
$$

Lemma B.15 (Weighted Jackson estimate; Freud et al., 1978, see also Lubinsky, 2007, §5). Let $f _ { \boldsymbol { \theta } } ( z ) = \mathrm { s i g n } ( z - \boldsymbol { \theta } )$ . There exists a universal constant $C > 0$ such that, for every $\theta \in \mathbb { R }$ and integer $p \geq 2$ , there is a polynomial $q$ of degree at most $p$ satisfying

$$
\| ( q - f _ { \theta } ) W \| _ { L ^ { 1 } } \leq { \frac { C } { \log ( p + 1 ) } } .
$$

Proof. Set $H _ { \theta } = f _ { \theta } W$ . By Freud et al. (1978, Theorem 2.2), taking $\delta = 1 / 2$ , there exists a polynomial q of degree at most p satisfying

$$
\| ( q - f _ { \theta } ) W \| _ { L ^ { 1 } } \leq C \left[ \operatorname* { s u p } _ { | h | \leq 1 / \log p } \| H _ { \theta } ( \cdot + h ) - H _ { \theta } \| _ { L ^ { 1 } } + \frac { \| H _ { \theta } \| _ { L ^ { 1 } } } { \log p } + \int _ { | z | \geq \sqrt { p } } | H _ { \theta } ( z ) | d z \right] .
$$

To bound the first term on the right hand side, first observe that $\| W ^ { \prime } \| _ { L ^ { 1 } } = 2$ . Using the fundamental theorem of calculus, we obtain $\| W ( \cdot + h ) - W \| _ { L ^ { 1 } } \leq | h | \| W ^ { \prime } \| _ { L ^ { 1 } } = 2 | h |$ . Next, $f _ { \theta } ( z + h )$ and $f _ { \boldsymbol { \theta } } ( z )$ difer only when z lies between $\theta - h$ and θ. On this interval, their diference has absolute value 2. Since $W \leq 1 , \| ( f _ { \theta } ( \cdot + h ) - f _ { \theta } ) W \| _ { L ^ { 1 } } \leq 2 | h |$ . Finally, adding and subtracting $f _ { \theta } ( z + h ) W ( z )$ and applying the triangle inequality gives

$$
\begin{array} { r } { \| H _ { \theta } ( \cdot + h ) - H _ { \theta } \| _ { L ^ { 1 } } \leq \| W ( \cdot + h ) - W \| _ { L ^ { 1 } } + \| ( f _ { \theta } ( \cdot + h ) - f _ { \theta } ) W \| _ { L ^ { 1 } } \leq 4 | h | . } \end{array}
$$

For the remaining terms, $| H _ { \theta } | = W$ gives $\| H _ { \theta } \| _ { L ^ { 1 } } = 2$ and $\begin{array} { r } { \int _ { | z | \geq \sqrt { p } } \left| H _ { \theta } ( z ) \right| d z = 2 e ^ { - \sqrt { p } } } \end{array}$ . Substituting these estimates yields $\| ( q - f _ { \theta } ) W \| _ { L ^ { 1 } } \leq C \left( 6 / \log p + 2 e ^ { - \sqrt { p } } \right) \leq C / \log ( p + 1 )$ . All constants are independent of $p$ and θ. □

## C Proofs for Improper Learning with Gaussian Marginals

In this section, we prove Theorem 2.1. We first establish the privacy guarantee of Algorithm 1.

Proposition C.1. Algorithm 1 satisfies $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ under replacement of one labeled example.

Proof. The feature map, the partition into blocks, and the algorithm parameters are data-independent. For each example $( x , y )$ , the loss $v \mapsto | \langle v , \bar { \psi } _ { p , T } ( x ) \rangle { - } y |$ is convex and T-Lipschitz, since $\Vert \bar { \psi } _ { p , T } ( x ) \Vert _ { 2 } \leq T$ As the optimization domain is the public unit ball, Lemma B.4 implies that each regression run is $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ with respect to its training block.

All blocks are disjoint, and running the DP gradient descent algorithm with the same privacy parameters independently on each block guarantees $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ by the guarantees of Lemma B.4 and parallel composition. Our output space being defined for the remainder of the computations via $\hat { v } _ { j } \mathrm { ' s }$ does not violate DP because of post-processing (Lemma B.3). Finally, applying the exponential mechanism (Lemma B.5) on a disjoint validation block of data satisfies $\varepsilon _ { \mathrm { D P } ^ { \mathrm { - } } } \mathrm { D P }$ . Therefore, by parallel composition, we obtain $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ □

Next, we establish a tail bound for the Hermite feature norm, which controls the bias introduced by clipping.

Lemma C.2 (Hermite feature norm tail bound). Let $d , p \geq 1$ be integers and $D _ { p } = { \binom { d + p } { p } }$ . Let $h _ { 1 } , \ldots , h _ { D _ { p } }$ enumerate the multivariate Hermite polynomials of degree at most $p ,$ orthonormal in $L ^ { 2 } ( \mathcal { N } ( 0 , I _ { d } ) )$ ). Define $\psi _ { p } ( x ) = ( h _ { 1 } ( x ) , \ldots , h _ { D _ { p } } ( x ) )$ . For every $\alpha \in ( 0 , 1 )$ satisfying $d \ge 2 p ^ { 2 } / \alpha$ and $X \sim { \mathcal { N } } ( 0 , I _ { d } )$

$$
\mathbb { P } \left[ \| \psi _ { p } ( X ) \| _ { 2 } > 2 \sqrt { D _ { p } } \right] \leq \alpha .
$$

Proof. Write $V = \| \psi _ { p } ( X ) \| _ { 2 } ^ { 2 } / D _ { p }$ . Orthonormality gives $\mathbb { E } [ V ] = 1$ . We claim that

$$
\operatorname { V a r } \left( V \right) \leq e ^ { 2 p ^ { 2 } / d } - 1 .\tag{13}
$$

Assuming this claim, Chebyshev’s inequality gives

$$
\begin{array} { r l r l } { \mathbb { P } \left[ \| \psi _ { p } ( X ) \| _ { 2 } > 2 \sqrt { D _ { p } } \right] = \mathbb { P } \left[ V - 1 > 3 \right] } \\ & { \leq \frac { \mathbb { E } \left[ \left( V - 1 \right) ^ { 2 } \right] } { 9 } } \\ & { \leq \frac { e ^ { 2 p ^ { 2 } / d } - 1 } { 9 } } \\ & { \leq \frac { e ^ { \alpha } - 1 } { 9 } \qquad } & & { \mathrm { ( s i n c e ~ } d \geq 2 p ^ { 2 } / \alpha \mathrm { ) } } \\ & { \leq \frac { ( e - 1 ) \alpha } { 9 } \qquad } & & { \mathrm { ( b y ~ c o n v e x i t y ~ o f ~ } e ^ { x } \mathrm { ~ f o r ~ } \alpha \in [ 0 , 1 ] \mathrm { ) } } \\ & { \leq \alpha . } \end{array}
$$

It remains to establish (13). For every integer $m \geq 0$ , define $\begin{array} { r } { K _ { m } ( x ) = \sum _ { | \kappa | \leq m } h _ { \kappa } ( x ) ^ { 2 } } \end{array}$ . In particular, $K _ { p } ( X ) \ : = \ : D _ { p } V$ . To compute the second moment of $K _ { p } ( X )$ , we use the generating function $\begin{array} { r } { F ( z , x ) = \sum _ { m = 0 } ^ { \infty } K _ { m } ( x ) z ^ { m } } \end{array}$ . Mehler’s formula (restated in Lemma B.9) gives

$$
F ( z , x ) = \frac { 1 } { 1 - z } \sum _ { \kappa \in \mathbb { N } _ { 0 } ^ { d } } z ^ { | \kappa | } h _ { \kappa } ( x ) ^ { 2 } = \frac { \exp { \left( \frac { z } { 1 + z } \left\| x \right\| _ { 2 } ^ { 2 } \right) } } { ( 1 - z ) ( 1 - z ^ { 2 } ) ^ { d / 2 } } .
$$

Since $\| X \| _ { 2 } ^ { 2 } \sim \chi _ { d } ^ { 2 }$ , its moment-generating function gives, for suficiently small $z , w > 0$

$$
\mathbb { E } [ F ( z , X ) F ( w , X ) ] = \frac { 1 } { ( 1 - z ) ^ { d + 1 } ( 1 - w ) ^ { d + 1 } } \left( 1 - \frac { 4 z w } { ( 1 - z ) ( 1 - w ) } \right) ^ { - d / 2 } .
$$

Expanding the last factor using the binomial series and comparing coeficients of $z ^ { p } w ^ { p }$ yields

$$
\mathbb { E } [ K _ { p } ( X ) ^ { 2 } ] = \sum _ { k = 0 } ^ { p } { \frac { 4 ^ { k } ( d / 2 ) _ { k } } { k ! } } { \binom { d + p } { p - k } } ^ { 2 } ,
$$

where $\begin{array} { r } { ( a ) _ { k } = \prod _ { j = 0 } ^ { k - 1 } ( a + j ) } \end{array}$ , with $( a ) _ { 0 } = 1$ . We therefore arrive at

$$
\begin{array} { r l } {  { \mathbb { E } [ V ^ { 2 } ] = \sum _ { k = 0 } ^ { p } \frac { 4 ^ { k } ( d / 2 ) _ { k } } { k ! } ( \frac { { \binom { d + p } { p - k } } } { D _ { p } } ) ^ { 2 } } } \\ & { \leq \displaystyle \sum _ { k = 0 } ^ { p } \frac { ( 2 p ^ { 2 } / d ) ^ { k } } { k ! } } \\ & { \leq e ^ { 2 p ^ { 2 } / d } , } \end{array}
$$

where the first inequality follows from $\begin{array} { r } { \binom { d + p } { p - k } / D _ { p } = \prod _ { j = 0 } ^ { k - 1 } \frac { p - j } { d + 1 + j } } \end{array}$ , together with $p - j \le p$ and $( d / 2 + j ) / ( d + 1 + j ) ^ { 2 } \leq 1 / ( 2 d )$ . Since $\mathbb { E } [ \bar { V } ] = 1 , ( 1 3 )$ follows. □

We now turn to the proof of Theorem 2.1. We use Algorithm 1 with the degree-p Hermite feature map $\psi _ { p }$ . Fix a universal constant $c \in ( 0 , 1 / 4 ]$ and a suficiently large universal constant C, depending only on c. Set

$$
p = \left\lceil C \alpha ^ { - 2 } \right\rceil , \quad D _ { p } = \binom { d + p } { p } , \quad T = 2 \sqrt { D _ { p } } , \quad k = \left\lceil \log _ { 2 } \frac { 2 } { \beta } \right\rceil , \quad M = \left\lceil \frac { C } { \alpha } \right\rceil .\tag{14a}
$$

Since $p \le ( C + 1 ) \alpha ^ { - 2 }$ , choosing $C _ { \mathrm { d i m } } \geq 2 ( C + 1 ) ^ { 2 } / c$ ensures that $d \geq C _ { \mathrm { d i m } } \alpha ^ { - 5 }$ implies $d \geq 2 p ^ { 2 } / ( c \alpha )$ Thus Lemma C.2 applies at level cα. Let $n _ { 0 } \geq 2$ be the smallest integer satisfying

$$
C \left( \frac { T \sqrt { D _ { p } \log ( 1 / \delta _ { \mathrm { D P } } ) } } { \varepsilon _ { \mathrm { D P } } n _ { 0 } } + \sqrt { \frac { D _ { p } } { n _ { 0 } } } \right) \leq c \alpha ,\tag{14b}
$$

and set

$$
m = \left\lceil C \left( \frac { 1 } { \alpha ^ { 2 } } + \frac { 1 } { \varepsilon _ { \mathrm { D P } } \alpha } \right) \log \frac { k M } { \beta } \right\rceil , \qquad n = k n _ { 0 } + m .\tag{14c}
$$

Proof of Theorem 2.1. Proposition C.1 establishes privacy. It remains to establish the accuracy, sample complexity, and running time bounds. Write $\mu = \mathcal { N } ( 0 , I _ { d } )$ , and recall the population regression loss $L _ { P } ( v ) \ = \ \mathbb { E } _ { ( X , Y ) \sim P } \left| \left. v , \bar { \psi } _ { p , T } ( X ) \right. - Y \right|$ , and its minimum $L ^ { \star } : = \mathrm { \ m i n } _ { \| v \| _ { 2 } \leq 1 } L _ { P } ( v )$ Additionally, for the training block $S _ { j }$ , write $\begin{array} { r } { \widehat { L } _ { S _ { j } } ( v ) : = n _ { 0 } ^ { - 1 } \sum _ { ( X _ { i } , Y _ { i } ) \in S _ { j } } \left| \left. v , \bar { \psi } _ { p , T } ( X _ { i } ) \right. - Y _ { i } \right| } \end{array}$ for the empirical regression loss.

We first bound the expected excess regression loss of a single training run. Orthonormality and clipping give $\mathbb { E } _ { \mu } \| \bar { \psi } _ { p , T } ( X ) \| _ { 2 } ^ { 2 } \leq D _ { p }$ and $\Vert \bar { \psi } _ { p , T } ( x ) \Vert _ { 2 } \leq T$ . The loss is convex and T-Lipschitz on all of $\mathbb { R } ^ { D _ { p } }$ , and the optimization domain is the public unit ball, of diameter 2. Fix a population minimizer $v ^ { \star } \in \mathrm { a r g m i n } _ { \| v \| _ { 2 } \leq 1 } L _ { P } ( v )$ , so that $L _ { P } ( v ^ { \star } ) = L ^ { \star }$ . Adding and subtracting $\widehat { L } _ { S _ { j } } ( \widehat { v } _ { j } )$ and $\widehat { L } _ { S _ { j } } ( v ^ { \star } )$ give

$$
\begin{array} { r l } & { L _ { P } ( \widehat { v } _ { j } ) - L ^ { \star } = \big [ L _ { P } ( \widehat { v } _ { j } ) - \widehat { L } _ { S _ { j } } ( \widehat { v } _ { j } ) \big ] + \big [ \widehat { L } _ { S _ { j } } ( \widehat { v } _ { j } ) - \widehat { L } _ { S _ { j } } ( v ^ { \star } ) \big ] + \big [ \widehat { L } _ { S _ { j } } ( v ^ { \star } ) - L ^ { \star } \big ] } \\ & { \qquad \leq \displaystyle \operatorname* { s u p } _ { \| v \| _ { 2 } \leq 1 } \big ( L _ { P } ( v ) - \widehat { L } _ { S _ { j } } ( v ) \big ) + \Big [ \widehat { L } _ { S _ { j } } ( \widehat { v } _ { j } ) - \displaystyle \operatorname* { m i n } _ { \| v \| _ { 2 } \leq 1 } \widehat { L } _ { S _ { j } } ( v ) \Big ] + \big [ \widehat { L } _ { S _ { j } } ( v ^ { \star } ) - L ^ { \star } \big ] . } \end{array}\tag{15}
$$

The inequality uses $\| \widehat { v } _ { j } \| _ { 2 } \leq 1$ . Since $v ^ { \star }$ is fixed and the examples in $S _ { j }$ are sampled from $P _ { \mathrm { : } }$ $\mathbb { E } _ { S _ { j } } \widehat { L } _ { S _ { j } } ( v ^ { \star } ) = L _ { P } ( v ^ { \star } ) = L ^ { \star }$ . Thus the final term in (15) has mean zero. Taking expectations over $S _ { j }$ and the randomness of the private optimizer, Lemmas B.4 and B.7 therefore give

$$
\mathbb { E } \big [ L _ { P } ( \widehat { v } _ { j } ) - L ^ { \star } \big ] \leq \mathbb { E } \Bigg [ \widehat { L } _ { S _ { j } } ( \widehat { v } _ { j } ) - \operatorname* { m i n } _ { \| v \| _ { 2 } \leq 1 } \widehat { L } _ { S _ { j } } ( v ) \Bigg ] + 2 \sqrt { \frac { D _ { p } } { n _ { 0 } } } \leq c \alpha ,\tag{16}
$$

where the last inequality follows from the choice of $n _ { 0 }$ in (14b).

We then use the standard confidence-amplification argument based on independent repetition; see, e.g., Shalev-Shwartz et al. (2010, proof of Theorem 26). By (16) and Markov’s inequality, $\mathbb { P } ( L _ { P } ( \widehat { v } _ { j } ) - L ^ { \star } > 2 c \alpha ) \ \leq \ 1 / 2$ . Since the $k$ training runs are independent, $\begin{array} { r } { \mathbb { P } ( \operatorname* { m i n } _ { j \in [ k ] } L _ { P } ( \widehat { v } _ { j } ) > } \end{array}$ $L ^ { \star } + 2 c \alpha ) \le 2 ^ { - k } \le \beta / 2$ . Thus, with probability at least $1 - \beta / 2$ 2

$$
\operatorname* { m i n } _ { j \in [ k ] } L _ { P } ( \widehat { v } _ { j } ) \leq L ^ { \star } + 2 c \alpha .\tag{17}
$$

Next, we argue about the correctness of the choice of our threshold, which requires our grid over [−1, 1] to be fine enough that the bias due to discretization is small, but not so fine that we blow up in terms of our sample complexity and running time when applying the exponential mechanism. If we select a threshold uniformly at random from the continuous interval [−1, 1], then Fact B.6 bounds the expected classification error under a uniform threshold $\tau \sim \mathrm { U n i f } [ - 1 , 1 ]$ by half the absolute loss. Then the expected 0-1 loss preserves our agnostic learning error requirements. However, rounding $\tau$ to the midpoint of its grid interval yields a threshold that is uniform over $\{ t _ { 1 } , \ldots , t _ { M } \}$ . At a fixed $x ,$ rounding changes the prediction only if the score $\langle \widehat { v } _ { j } , \bar { \psi } _ { p , T } ( x ) \rangle$ lies between $\tau$ and the midpoint $\tau$ is rounded $^ { \mathrm { t o , } }$ which happens with probability at most $1 / ( 2 M )$ . Hence, the expectation of the population error of $\widehat { v } _ { j }$ over the random sampling from the discretized grid is $\begin{array} { r } { M ^ { - 1 } \sum _ { \ell = 1 } ^ { M } \mathrm { e r r } _ { P } ( h _ { j , \ell } ) \le L _ { P } ( \widehat { v } _ { j } ) / 2 + 1 / ( 2 M ) } \end{array}$ . Since min<sub>ℓ</sub> err $\begin{array} { r } { { P } ( h _ { j , \ell } ) \mathop { \leq } M ^ { - 1 } \mathop { \sum } _ { \ell = 1 } ^ { M } \mathop { \operatorname { e r r } } _ { P } ( h _ { j , \ell } ) } \end{array}$ ， conditioned on the event in (17) and taking the minimum over all $\widehat { v } _ { j } \mathrm { ^ { s , } }$ we have

$$
\operatorname* { m i n } _ { j , \ell } \mathrm { e r r } _ { P } ( h _ { j , \ell } ) \leq \frac { 1 } { 2 } \operatorname* { m i n } _ { j \in [ k ] } L _ { P } ( \widehat { v } _ { j } ) + \frac { 1 } { 2 M } \leq \frac { 1 } { 2 } L ^ { \star } + 2 c \alpha .\tag{18}
$$

Let h denote the classifier selected by the exponential mechanism on $S _ { \mathrm { v a l } }$ . Conditional on the training outputs, Lemma B.5, with $K = k M , \eta = \beta / 2$ , and m as in (14c), gives $\mathrm { e r r } _ { P } ( h ) \leq$ mi $1 _ { j , \ell } \mathrm { e r r } _ { P } ( h _ { j , \ell } )$ + cα with probability at least $1 - \beta / 2$ . Thus, with probability at least $1 - \beta$

$$
\mathrm { e r r } _ { P } ( h ) \leq L ^ { \star } / 2 + 3 c \alpha ,\tag{19}
$$

which comes due to sampling and the noise due to the exponential mechanism.

We next compare $L ^ { \star }$ with the error of the best halfspace. Fix $c _ { 0 } \in { \mathcal { C } }$ . By Lemma B.10, and the choice of the degree p in (14a), there is a polynomial $q = \langle v _ { q } , \psi _ { p } \rangle$ with $\| v _ { q } \| _ { 2 } \le 1$ and $\Vert q - c _ { 0 } \Vert _ { L ^ { 1 } ( \mu ) } \leq c \alpha$ Write $s ( x ) : = T /$ max $\{ T , \| \psi _ { p } ( x ) \| _ { 2 } \}$ , so that $\bar { \psi } _ { p , T } ( x ) \ : = \ : s ( x ) \psi _ { p } ( x )$ . Since $0 < s ( x ) \leq \ " 1$ and $| y | = 1$ ，

$$
\begin{array} { r l } & { \left| \left. v _ { q } , \bar { \psi } _ { p , T } ( x ) \right. - y \right| = | s ( x ) q ( x ) - y | } \\ & { \qquad \leq s ( x ) | q ( x ) - y | + 1 - s ( x ) } \\ & { \qquad \leq | q ( x ) - c _ { 0 } ( x ) | + | c _ { 0 } ( x ) - y | + 1 - s ( x ) . } \end{array}
$$

Taking expectations over a population example $( X , Y ) \sim P$ yields

$$
L _ { P } ( v _ { q } ) \leq 2 \operatorname { e r r } _ { P } ( c _ { 0 } ) + \| q - c _ { 0 } \| _ { L ^ { 1 } ( \mu ) } + \mathbb { E } _ { \mu } [ 1 - s ( X ) ] .\tag{20}
$$

Moreover, since $0 \leq 1 - s ( x ) \leq \mathbf { 1 } \{ \| \psi _ { p } ( x ) \| _ { 2 } > T \}$ , Lemma C.2, applied at level cα, gives $\mathbb { E } _ { \mu } [ 1 - s ( X ) ] \leq$ $\operatorname* { P r } _ { \mu } ( \| \psi _ { p } ( X ) \| _ { 2 } > T ) \leq c \alpha$ . Since $\| v _ { q } \| _ { 2 } \le 1$ , we conclude that $L ^ { \star } \leq L _ { P } ( v _ { q } ) \leq 2 \mathrm { e r r } _ { P } ( c _ { 0 } ) + 2 c \alpha$ Taking the infimum over $c _ { 0 } \in \mathcal { C }$ gives

$$
L ^ { \star } \leq 2 \mathrm { O P T } + 2 c \alpha .\tag{21}
$$

Combining (19) and (21), we obtain the desired

$$
\mathrm { e r r } _ { P } ( h ) \leq \mathrm { O P T } + 4 c \alpha \leq \mathrm { O P T } + \alpha ,
$$

where the last inequality uses $c \leq 1 / 4$

The choices in (14b) and (14c) satisfy

$$
\eta _ { 0 } = \widetilde { \mathcal { O } } \left( \frac { D _ { p } } { \alpha ^ { 2 } } + \frac { D _ { p } } { \varepsilon _ { \mathrm { D P } } \alpha } \right) , \qquad m = \widetilde { \mathcal { O } } \left( \left( \frac { 1 } { \alpha ^ { 2 } } + \frac { 1 } { \varepsilon _ { \mathrm { D P } } \alpha } \right) \log \frac { 2 } { \beta } \right) ,
$$

where the bound on $n _ { 0 }$ uses $T \sqrt { D _ { p } } = 2 D _ { p }$ . Since $k = \mathcal { O } \left( \log ( 2 / \beta ) \right)$ and $n = k n _ { 0 } + m$ , this yields the stated sample bound on $n .$

Finally, we analyse the running time. Computing the clipped Hermite features requires $\mathcal { O } \left( p D _ { p } \right)$ arithmetic operations per example. Given these features, subgradient evaluation and projection onto the Euclidean unit ball each take $\mathcal { O } \left( D _ { p } \right)$ time, so Lemma B.4 gives a total cost of $\mathcal { O } \left( k n _ { 0 } ^ { 2 } D _ { p } \right)$ for the k private ERM runs. Evaluating the kM candidate classifiers on the validation sample requires $\mathcal { O } \left( k M m D _ { p } \right)$ operations. Thus, the overall running time is

$$
\mathcal { O } \left( n p D _ { p } + k n _ { 0 } ^ { 2 } D _ { p } + k M m D _ { p } \right) = \mathrm { p o l y } \left( n \right) ,
$$

since $M \leq p < D _ { p } \leq n _ { 0 }$ and $k n _ { 0 } + m = n$

## D Private Improper Agnostic Learning: Log-Concave Marginals

We extend the learner of Section 2 to known, full-dimensional<sup>1</sup> log-concave marginals. The regression, confidence amplification, thresholding, and private validation steps are unchanged. We replace the Hermite features by a polynomial basis that is orthonormal under the marginal. The approximation guarantee in Lemma D.3 allows us to restrict regression to a coeficient ball of constant radius. We use the known marginal to choose a clipping threshold $T$ that controls the additional regression loss while keeping the feature norms bounded.

Theorem D.1 (DP improper agnostic learner for log-concave marginals). Let $\alpha , \beta \in ( 0 , 1 )$ and $ { \varepsilon } _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } \in ( 0 , 1 / 2 ]$ . Set $p = \lceil \exp ( C / \alpha ) \rceil$ for a suficiently large universal constant $C > 0$ , let $D _ { p } = { \binom { d + p } { p } }$ , and let T be as in (22). Let P be any distribution on $\mathbb { R } ^ { d } \times \{ \pm 1 \}$ with a known, fulldimensional log-concave marginal $\mu$ . Then the modification of Algorithm 1 described in Appendix D.1 is $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) { - } D P$ and, given

$$
n = \widetilde { \mathcal { O } } \left( \frac { D _ { p } } { \alpha ^ { 2 } } + \frac { T \sqrt { D _ { p } } } { \varepsilon _ { \mathrm { D P } } \alpha } \right)
$$

i.i.d. samples from $P ,$ , returns a classifier h satisfying e $\cdot \mathrm { r } _ { P } ( h ) \leq \mathrm { O P T } + \alpha$ with probability at least $1 - \beta$ . The algorithm runs in time poly(n), provided that constructing the orthonormal feature map, evaluating it at an input, and computing T from public information each take time ${ \mathrm { p o l y } } ( D _ { p } )$

As we show in Remark 3, $T = \sqrt { D _ { p } / ( c \alpha ) }$ is an admissible clipping threshold. Consequently, the sample bound is at most $\widetilde { \mathcal { O } } \left( D _ { p } / \alpha ^ { 2 } + D _ { p } / ( \varepsilon _ { \mathrm { D P } } \alpha ^ { 3 / 2 } ) \right)$

## D.1 The Algorithm

Let $\mu$ be a known, full-dimensional log-concave distribution on $\mathbb { R } ^ { d }$ . We modify Algorithm 1 by replacing the Hermite features with a polynomial basis orthonormal under $\mu ,$ choosing a clipping threshold adapted to $\mu ,$ and changing the optimization domain. We increase the polynomial degree to achieve the required approximation accuracy. We keep $k , M$ , and m as in (14a) and (14c), and replace the block size $n _ { 0 }$ by (27). The private optimization procedure, independent repetitions, thresholding, and private validation remain unchanged. We discuss the changes to Algorithm 1 in greater detail in the remainder of this section.

First, for the existence of a polynomial with good approximating guarantees, we increase the degree $p = \lceil \exp ( C / \alpha ) \rceil$ for a suficiently large universal constant $C > 0$ . For the optimization domain, we replace the unit coeficient ball in Algorithm 1 with $B _ { 2 } ^ { D _ { p } } ( R _ { 0 } ) : = \{ v \in \mathbb { R } ^ { D _ { p } } : \| v \| _ { 2 } \leq R _ { 0 } \}$ where $R _ { 0 }$ is the universal constant in Lemma D.3.

Next, we recall how to construct an orthonormal basis for the polynomials of degree at most $p$ in $L ^ { 2 } ( \mu )$ . Let $m _ { p } ( x ) = ( x ^ { \kappa } ) _ { | \kappa | \leq p }$ be the vector of monomials of degree at most $p .$ Following the standard moment-matrix orthonormalization (Dunkl and Xu, 2014), define the population Gram matrix $G _ { p } = \mathbb { E } _ { \mu } [ m _ { p } ( X ) m _ { p } ( X ) ^ { \top } ]$ and set $\psi _ { p } ( x ) = G _ { p } ^ { - 1 / 2 } m _ { p } ( x )$ . The entries of $G _ { p }$ are the moments of $\mu$ through degree $2 p .$ . Log-concavity ensures that all moments are finite, while full dimensionality guarantees that $G _ { p }$ is positive definite. Since $G _ { p } ^ { - 1 / 2 }$ is invertible, the coordinates of $\psi _ { p }$ span the polynomials of degree at most $p .$ Moreover, $\mathbb { E } _ { \mu } [ \dot { \psi _ { p } } ( X ) \psi _ { p } ( X ) ^ { \top } ] = I _ { D _ { p } }$ , so they form an orthonormal basis of this polynomial space in $L ^ { 2 } ( \mu )$ . In particular, $\| \langle v , \psi _ { p } \rangle \| _ { L ^ { 2 } ( \mu ) } = \| v \| _ { 2 }$ and $\mathbb { E } _ { \mu } \| \psi _ { p } ( X ) \| _ { 2 } ^ { 2 } = D _ { p }$ This construction based on the moment-matrix provides a general way to obtain the required orthonormal basis. When an explicit basis is available, it may be used directly without forming $G _ { p }$ provided its construction and evaluation satisfy the computational assumption in Theorem D.1. For example, the normalized Hermite basis for the standard Gaussian marginal can be constructed and evaluated in $\mathrm { p o l y } ( D _ { p } )$ arithmetic operations.

Finally, we discuss clipping. As in the Gaussian case, we radially clip the feature vectors to obtain a uniform bound on their norms: $\bar { \psi } _ { p , T } ( x ) : = T \psi _ { p } ( x ) /$ max $\{ T , \| \psi _ { p } ( x ) \| _ { 2 } \}$ . Since $\mu$ is known, we can choose the clipping threshold directly from the marginal. For a suficiently small universal constant $c > 0$ , set

$$
T = \operatorname* { m i n } \left\{ t > 0 : \mathbb { E } _ { \mu } \left[ \left( 1 - \frac { t } { \| \psi _ { p } ( X ) \| _ { 2 } } \right) _ { + } \right] \leq c \alpha \right\} ,\tag{22}
$$

where $( x ) _ { + } = \operatorname* { m a x } \{ x , 0 \}$ . The argument for (7) uses only orthonormality and therefore also applies to the present feature map. Since $( 1 - t / \| \psi _ { p } ( x ) \| _ { 2 } ) _ { + } \leq \mathbf { 1 } \{ \| \psi _ { p } ( x ) \| _ { 2 } > t \}$ , the choice $t = \sqrt { D _ { p } / ( c \alpha ) }$ is admissible, and hence $T \leq \sqrt { D _ { p } / ( c \alpha ) }$ . Depending on the marginal and the parameter regime, marginal-dependent choices may yield a smaller threshold. For example, if $\mu = N ( 0 , I _ { d } )$ and $d \geq 2 p ^ { 2 } / ( c \alpha )$ , Lemma C.2 shows that $T \leq 2 { \sqrt { D _ { p } } } ,$ so the privacy term improves to $\mathcal { O } \left( D _ { p } / ( \varepsilon _ { \mathrm { D P } } \alpha ) \right) . ^ { 2 }$

## D.2 Analysis

In this section, we prove Theorem D.1. We first establish privacy.

Proposition D.2. The learner in Theorem D.1 satisfies $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ under replacement of one labeled example.

Proof. The feature map, clipping threshold, coeficient ball, and sample partition are public. Since $\Vert \bar { \psi } _ { p , T } ( x ) \Vert _ { 2 } \leq T$ , the loss is convex and T-Lipschitz in v on all of $\mathbb { R } ^ { D _ { p } }$ . Thus, each regression run is $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ by Lemma B.4. The neighboring-dataset argument in the proof of Proposition C.1 then applies unchanged, establishing $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ for the full learner. □

We next prove the bounded-norm polynomial approximation guarantee. The key ingredient is the exponential density bound for one-dimensional log-concave distributions (Fact B.13), which allows us to transfer weighted polynomial approximation bounds to the $L ^ { 1 }$ and $L ^ { 2 }$ norms under the marginal. The result extends to arbitrary halfspaces by considering their one-dimensional projections.

Lemma D.3 (Bounded-norm polynomial approximation of halfspaces under log-concave measures). There exist universal constants $C , R _ { 0 } \geq 1$ such that the following holds. Let $\mu$ be a log-concave probability measure on $\mathbb { R } ^ { d }$ . For every integer $p \geq 2$ and halfspace $c : \mathbb { R } ^ { d }  \{ - 1 , 1 \}$ , there is a polynomial $q$ of degree at most p satisfying $\| q - c \| _ { L ^ { 1 } ( \mu ) } \leq C / \log ( p + 1 )$ and $\| q \| _ { L ^ { 2 } ( \mu ) } \leq R _ { 0 }$ . In particular, taking $p = \lceil \exp ( C / \alpha ) \rceil$ gives $\| q - c \| _ { L ^ { 1 } ( \mu ) } \leq \dot { \alpha }$

Proof. Consider a log-concave probability measure ν on R with mean zero and variance one, and set $f _ { \theta } \left( z \right) = \mathrm { s i g n } ( z - \theta )$ . We will first prove that there exists a polynomial r of degree at most $p$ satisfying

$$
\| r - f _ { \theta } \| _ { L ^ { 1 } ( \nu ) } \leq C / \log ( p + 1 ) \mathrm { a n d } \| r - f _ { \theta } \| _ { L ^ { 2 } ( \nu ) } ^ { 2 } \leq C .\tag{23}
$$

Assuming the above is true, we next extend it to $\mathbb { R } ^ { d }$ . Write the halfspace $c \left( x \right) = \mathrm { s i g n } ( \left. u , x \right. - \theta )$ For $X \sim \mu$ , consider the mean $a = \mathbb { E } _ { \mu } \left. u , X \right.$ and variance $b ^ { 2 } = \operatorname { V a r } _ { \mu } \left( \langle u , X \rangle \right) > 0 . ^ { 3 }$ Let ν denote the distribution of the normalized random variable $Z = ( \langle u , X \rangle - a ) / b$ . Then ν is log-concave, with mean zero and variance one. Apply Equation (23) to $f _ { ( \theta - a ) / b }$ and set $q ( x ) = r \left( \left( \left. u , x \right. - a \right) / b \right)$ Since $q ( X ) = r ( Z )$ and $c ( X ) = f _ { ( \theta - a ) / b } ( Z )$ , this preserves the degree of the polynomial and yields $\| q - c \| _ { L ^ { 1 } ( \mu ) } \leq C / \log ( p + 1 )$ and $\| q - c \| _ { L ^ { 2 } ( \mu ) } ^ { 2 } \leq C$ . Since $\| c \| _ { L ^ { 2 } ( \mu ) } = 1$ , using the triangle inequality on $q = ( q - c ) + c \ : \mathrm { g i }$ ives

$$
\| q \| _ { L ^ { 2 } ( \mu ) } \leq \| q - c \| _ { L ^ { 2 } ( \mu ) } + \| c \| _ { L ^ { 2 } ( \mu ) } \leq \sqrt { C } + 1 = : R _ { 0 } .
$$

It remains to prove Equation (23). Let $W \left( z \right) = \exp \left( - \left| z \right| / 3 2 \right)$ Then using Kalai et al. (2008, Lemma 6) (restated in Fact B.13), the density satisfies

$$
g ( z ) \leq e \exp ( - | z | / 1 6 ) .\tag{24}
$$

Note that $f _ { \theta } \left( z \right) W \left( z \right)$ is the product of a sign function and an exponentially decaying weight, so $| f _ { \theta } ( z ) W ( z ) | = e ^ { - | z | / 3 2 }$ . Then, the approximation theorem of Freud et al. (1978) (restated in Lemma B.15) gives a polynomial r of degree at most p satisfying

$$
\| ( r - f _ { \theta } ) W \| _ { L ^ { 1 } } \leq { \frac { C } { \log ( p + 1 ) } } .\tag{25}
$$

We now bound the $L _ { 2 }$ error:

$$
\begin{array} { r l r } {  { \| ( r - f _ { \theta } ) W \| _ { L ^ { 2 } } ^ { 2 } \leq \| ( r - f _ { \theta } ) W \| _ { L ^ { \infty } } \| ( r - f _ { \theta } ) W \| _ { L ^ { 1 } } \leq ( \| r W \| _ { L ^ { \infty } } + \| f _ { \theta } W \| _ { L ^ { \infty } } ) \frac { C } { \log ( p + 1 ) } } } \\ & { } & { \leq C \log ( p + 1 ) \frac { C } { \log ( p + 1 ) } \leq C . ~ } \end{array}\tag{26}
$$

Here the first inequality follows from Hölder’s inequality and the second uses Equation (25) and triangle inequality. The third inequality follows from Mhaskar and Saf (1984, Theorem 3.1) (restated in Lemma B.14) and observing that $\| f _ { \theta } W \| _ { L ^ { \infty } } \leq 1$ . Now using $g \le e W ^ { 2 }$ and $W \leq 1$ we can transfer Equations (25) and (26) to a norm with respect to measure induced by $\nu ,$ to get Equation (23). This completes the proof. □

We now turn to the proof of Theorem D.1. Fix $c \in ( 0 , 1 / 4 ]$ as in (22). The block-size condition difers from the Gaussian case only through the radius $R _ { 0 }$ and the threshold $T \colon$ let $n _ { 0 } \geq 2$ be the smallest integer satisfying

$$
C \left[ \frac { R _ { 0 } T \sqrt { D _ { p } \log ( 1 / \delta _ { \mathrm { D P } } ) } } { \varepsilon _ { \mathrm { D P } } n _ { 0 } } + R _ { 0 } \sqrt { \frac { D _ { p } } { n _ { 0 } } } \right] \leq c \alpha .\tag{27}
$$

Proof of Theorem D.1. Proposition D.2 provides the privacy guarantee. We now prove the accuracy and running time bounds. Write $L _ { P } ( v ) = \mathbb { E } _ { ( X , Y ) \sim P } | \langle v , \bar { \psi } _ { p , T } ( X ) \rangle - Y |$ and $\begin{array} { r } { L ^ { \star } = \operatorname* { m i n } _ { \| v \| _ { 2 } \leq R _ { 0 } } L _ { P } ( v ) } \end{array}$ The minimum exists since the domain is compact and the clipped loss is continuous in v.

We first compare $L ^ { \star }$ with the error of the best halfspace. Fix $c _ { 0 } \in \mathcal { C }$ . By Lemma D.3, our choice of p gives a polynomial $q = \langle v _ { q } , \psi _ { p } \rangle$ such that $\| q - c _ { 0 } \| _ { L ^ { 1 } ( \mu ) } \leq$ cα and $\| v _ { q } \| _ { 2 } = \| q \| _ { L ^ { 2 } ( \mu ) } \leq R _ { 0 }$ . Let $s ( x ) = T / \operatorname* { m a x } \{ T , \| \psi _ { p } ( x ) \| _ { 2 } \}$ . Since $1 - s ( x ) = ( 1 - T / \| \ddot { \psi } _ { p } ( x ) \| _ { 2 } ) _ { + } , ( 2 2 )$ gives $\mathbb { E } _ { \mu } [ 1 - \overset { \cdot } { s } ( X ) ] \leq c \alpha$ Substituting this into (20) gives $L _ { P } ( v _ { q } ) \leq 2 \mathrm { e r r } _ { P } ( c _ { 0 } ) + 2 c \alpha$ . Since $\| v _ { q } \| _ { 2 } \le R _ { 0 }$ , taking the infimum over $c _ { 0 } \in \mathcal { C }$ yields

$$
L ^ { \star } \leq 2 \mathrm { O P T } + 2 c \alpha .\tag{28}
$$

Next, we bound the expected excess regression loss of a single training run. The loss is convex and T-Lipschitz on all of R $D _ { p }$ , the optimization domain has diameter $2 R _ { 0 }$ , and $\mathbb { E } _ { \mu } \| \bar { \psi } _ { p , T } ( X ) \| _ { 2 } ^ { 2 } \leq D _ { p }$ Applying the decomposition in (15), we obtain

$$
\begin{array} { r l } & { \mathbb { E } [ L _ { P } ( \widehat { v } _ { j } ) - L ^ { \star } ] \leq \mathbb { E } \bigg [ \widehat { L } _ { S _ { j } } ( \widehat { v } _ { j } ) - \underset { \| v \| _ { 2 } \leq R _ { 0 } } { \operatorname* { m i n } } \widehat { L } _ { S _ { j } } ( v ) \bigg ] + \mathbb { E } \underset { \| v \| _ { 2 } \leq R _ { 0 } } { \operatorname* { s u p } } \Big ( L _ { P } ( v ) - \widehat { L } _ { S _ { j } } ( v ) \Big ) } \\ & { \qquad \leq C \frac { R _ { 0 } T \sqrt { D _ { p } \log \left( 1 / \delta _ { \mathrm { D P } } \right) } } { \varepsilon _ { \mathrm { D P } } n _ { 0 } } + 2 R _ { 0 } \sqrt { \frac { D _ { p } } { n _ { 0 } } } \leq c \alpha . } \end{array}
$$

where the second inequality follows from Lemmas B.4 and B.7, and the last from (27).

Since the training runs are independent and the thresholding and validation steps use the same choices of $k , M ,$ , and $m _ { : }$ , the argument leading to (19) applies with $\begin{array} { r } { L ^ { \star } = \operatorname* { m i n } _ { \| v \| _ { 2 } \leq R _ { 0 } } L _ { P } ( v ) } \end{array}$ Consequently, using $( 2 8 ) , \mathrm { e r r } _ { P } ( h ) \leq \mathrm { O P T } + \alpha$ with probability at least $1 - \beta .$

It remains to bound the sample size and running time. Since $R _ { 0 }$ is a universal constant, (27) gives

$$
n _ { 0 } = O \left( \frac { D _ { p } } { \alpha ^ { 2 } } + \frac { T \sqrt { D _ { p } \log ( 1 / \delta _ { \mathrm { D P } } ) } } { \varepsilon _ { \mathrm { D P } } \alpha } \right) ,
$$

independent of $\beta ,$ , and $k = \mathcal { O } \left( \log ( 2 / \beta ) \right)$ . The validation size m is absorbed by this bound. Indeed, $D _ { p } \geq 1$ , and every point satisfies $\| \psi _ { p } ( x ) \| _ { 2 } \geq 1$ because the constant polynomial has unit $L ^ { 2 } ( \mu )$ norm. Hence, the expectation in (22) is at least $( 1 - t ) _ { + } ,$ so $T \geq 1 - c \alpha \geq 3 / 4$ . This proves the stated bound on $n = k n _ { 0 } + m$ . Since $T \leq \sqrt { D _ { p } / ( c \alpha ) }$ by (7), the bound is always at most $\widetilde { \mathcal { O } } \left( D _ { p } / \alpha ^ { 2 } + D _ { p } / ( \varepsilon _ { \mathrm { D P } } \alpha ^ { 3 / 2 } ) \right)$

Finally, we bound the running time, measured in arithmetic operations. By the computational hypothesis in Theorem D.1, computing $G _ { p } ^ { - 1 / 2 }$ and $T$ takes $\mathrm { p o l y } ( D _ { p } )$ operations. For each example, evaluating $\psi _ { p } ( x ) = G _ { p } ^ { - 1 / 2 } m _ { p } ( x )$ takes $\mathcal { O } \left( p D _ { p } \right)$ operations for the monomials and $\mathcal { O } \left( D _ { p } ^ { 2 } \right)$ for the matrix–vector product, and clipping takes $\mathcal { O } \left( D _ { p } \right)$ more. The remaining steps are those of Algorithm 1 with the unit ball replaced by $B _ { 2 } ^ { D _ { p } } ( R _ { 0 } )$ . Projection onto this ball still takes $\mathcal { O } \left( D _ { p } \right)$ operations, so, as in the proof of Theorem 2.1, the private ERM runs and the validation step take $\mathcal { O } \left( k n _ { 0 } ^ { 2 } D _ { p } \right)$ and $\mathcal { O } \left( k M m D _ { p } \right)$ operations, respectively. Since $p < D _ { p }$ , the total is

$$
\mathrm { p o l y } ( D _ { p } ) + \mathcal { O } \left( n D _ { p } ^ { 2 } + k n _ { 0 } ^ { 2 } D _ { p } + k M m D _ { p } \right) = \mathrm { p o l y } ( n ) ,
$$

where the last equality uses $M \leq p < D _ { p } \leq n _ { 0 }$ and $k n _ { 0 } + m = n$

## E Proofs for Proper Learning with Gaussian Marginals

Proof of Lemma 3.2. We follow the subspace construction in the proof of Diakonikolas et al. (2021a, Proposition 5). Write $\mu = \mathcal { N } ( 0 , I _ { d } )$ . Let $q ^ { * }$ be the orthogonal projection of $r ( x ) = \mathbb { E } [ U \mid X = x ]$ onto $\mathcal { P } _ { p }$ . The squared-loss identity gives $\| q - q ^ { * } \| _ { L ^ { 2 } ( \mu ) } \leq c _ { 0 } \alpha$ . Since $| r | \leq 1$ , we also have $\| q ^ { * } \| _ { L ^ { 2 } ( \mu ) } \leq 1$ and $\| q \| _ { L ^ { 2 } ( \mu ) } \leq 1 + c _ { 0 } \alpha$ . Consequently, for every $Q \in \mathcal { P } _ { p }$

$$
\begin{array} { r } { | \mathbb { E } \left[ ( U - q ( X ) ) Q ( X ) \right] | = | \mathbb { E } \left[ ( q ^ { * } ( X ) - q ( X ) ) Q ( X ) \right] | \leq c _ { 0 } \alpha \left\| Q \right\| _ { L ^ { 2 } ( \mu ) } . } \end{array}\tag{29}
$$

Set $M = \mathbb { E } _ { \mu } \left[ \nabla q ( X ) \nabla q ( X ) ^ { \top } \right]$ , and let V be the span of its eigenvectors with eigenvalues at least $\alpha ^ { 2 } / C _ { 0 }$ . The Hermite derivative identity and trace estimate in Diakonikolas et al. (2021a, Fact 19 and the proof of Lemma 6) give t $\operatorname { r } ( M ) \leq p \left\| q \right\| _ { L ^ { 2 } ( \mu ) } ^ { 2 }$ . It follows that

$$
\dim ( V ) \leq { \frac { C _ { 0 } p ( 1 + c _ { 0 } \alpha ) ^ { 2 } } { \alpha ^ { 2 } } } = { \mathcal { O } } \left( \alpha ^ { - 4 } \right) .
$$

Fix a halfspace $f = h _ { w , t }$ and write $w = w _ { V } + w _ { \perp }$ for its orthogonal decomposition with respect to V. If $w _ { \perp } = 0$ , then f already has its normal in $V$ . Otherwise, set $\xi = w _ { \perp } / \parallel w _ { \perp } \parallel _ { 2 }$ , and define

$$
( A _ { \xi } g ) ( x ) = \mathbb { E } _ { G \sim \mathcal { N } ( 0 , 1 ) } \left[ g \left( x - \left. \xi , x \right. \xi + G \xi \right) \right] .
$$

Under $\mu ,$ this is conditional expectation given the projection onto $\xi ^ { \perp }$ . Thus, $A _ { \xi }$ is an orthogonal projection in $L ^ { 2 } ( \mu )$ , a contraction in $L ^ { 1 } ( \mu )$ , and preserves $\mathcal { P } _ { p }$ . Applying Diakonikolas et al. (2021a, Claim 10) directly gives

$$
\begin{array} { r } { \left. q - A _ { \xi } q \right. _ { L ^ { 2 } ( \mu ) } ^ { 2 } \le \operatorname { \mathbb { E } } \left[ ( \partial _ { \xi } q ( X ) ) ^ { 2 } \right] = \xi ^ { \top } M \xi \le \alpha ^ { 2 } / C _ { 0 } . } \end{array}\tag{30}
$$

By Lemma B.10, taking $C _ { 0 }$ suficiently large gives $a \in \mathcal { P } _ { p }$ with $\| a \| _ { L ^ { 2 } ( \mu ) } \leq 1$ and $\| f - a \| _ { L ^ { 1 } ( \mu ) } \leq \alpha$ Set $Q = a - A _ { \xi } a$ . Then $\| Q \| _ { L ^ { 2 } ( \mu ) } \leq 1 , A _ { \xi } Q = 0$ , and

$$
\begin{array} { r } { \| ( f - A _ { \xi } f ) - Q \| _ { L ^ { 1 } ( \mu ) } \leq \| f - a \| _ { L ^ { 1 } ( \mu ) } + \| A _ { \xi } ( f - a ) \| _ { L ^ { 1 } ( \mu ) } \leq 2 \alpha . } \end{array}
$$

Since $| U | \le 1$ and $A _ { \xi } Q = 0$ , we obtain

$$
\begin{array} { r l } & { \mathbb { E } \left[ U ( f ( X ) - A _ { \xi } f ( X ) ) \right] \leq 2 \alpha + \mathbb { E } \left[ ( U - q ( X ) ) Q ( X ) \right] + \mathbb { E } \left[ ( q ( X ) - A _ { \xi } q ( X ) ) Q ( X ) \right] } \\ & { \qquad \leq ( 2 + c _ { 0 } + C _ { 0 } ^ { - 1 / 2 } ) \alpha , } \end{array}
$$

where the last inequality uses Equations (29) and (30). Now,

$$
A _ { \xi } f ( x ) = \mathbb { E } _ { G } \left[ \mathrm { s i g n } ( \langle w _ { V } , x \rangle + \| w _ { \perp } \| _ { 2 } G + t ) \right] .
$$

This is a mixture of halfspaces with unit normals in $V ,$ , or constant classifiers if $w _ { V } = 0$ . Hence one such classifier $h ^ { * }$ satisfies $\mathbb { E } [ U ( f ( X ) - h ^ { * } ( X ) ) ] \leq ( 2 + c _ { 0 } + C _ { 0 } ^ { - 1 / 2 } ) \alpha$ . When $w _ { \perp } = 0$ , take $h ^ { * } = f$

For the finite class, use the covering and discretization bounds of Diakonikolas et al. (2021a, Fact 21 and Fact 22 with its proof), restated in Facts B.11 and B.12. Normalize the nonzero centers of an $\alpha / 2$ -net of the unit ball of V to obtain an α-net V of its unit sphere. Set

$$
R = \alpha \left\lceil K _ { 0 } \sqrt { \log ( 1 / \alpha ) } / \alpha \right\rceil
$$

for a suficiently large universal constant $K _ { 0 }$ . Define

$$
\begin{array} { r } { \mathcal { H } ( q ) = \{ h _ { v , j \alpha } : v \in \mathcal { V } , \ j \in \mathbb { Z } , \ | j \alpha | \leq R \} \cup \{ h _ { e _ { 1 } , R } , h _ { e _ { 1 } , - R } \} . } \end{array}
$$

When $V = \{ 0 \}$ , omit the first set. The two additional halfspaces approximate the constant classifiers. The discretization bound gives an $h \in \mathcal { H } ( q )$ satisfying $\mathbb { P } [ h ( X ) \neq h ^ { * } ( X ) ] = \mathcal { O } \left( \alpha \right)$ , including when $h ^ { * }$ is constant or has threshold outside $[ - R , R ]$ . Thus,

$$
\begin{array} { r } { \mathbb { E } \left[ U ( f ( X ) - h ( X ) ) \right] \leq \mathbb { E } \left[ U ( f ( X ) - h ^ { * } ( X ) ) \right] + 2 \mathbb { P } [ h ( X ) \neq h ^ { * } ( X ) ] \leq C \alpha . } \end{array}
$$

Finally, the covering bound gives

$$
\log \left| \mathcal { H } ( q ) \right| = \mathcal { O } \left( ( \mathrm { d i m } ( V ) + 1 ) \log ( 1 / \alpha ) \right) = \mathcal { O } \left( \alpha ^ { - 4 } \log ( 1 / \alpha ) \right) .
$$

The construction uses only q and α.

Proof of Lemma $\ 3 . 4 \cdot$ . Let

$$
\widehat { m } = \frac { 2 } { n } \sum _ { ( X _ { i } , Y _ { i } ) \in S _ { 1 } } Y _ { i } \bar { \psi } _ { p , T } ( X _ { i } ) .
$$

Each summand has norm at most $T ,$ , so $\widehat { m }$ has replacement sensitivity $4 T / n$ . The noise scale in Algorithm 2 therefore makes $\widehat { m } + Z \ ( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) \ – \mathrm { D P }$ by standard Gaussian mechanism (Dwork and Roth, 2014). Projection and construction of q are post-processing and preserve privacy.

Write $\ell = \log ( 8 / \beta )$ and $L _ { \delta _ { \mathrm { D P } } } = \log ( 1 . 2 5 / \delta _ { \mathrm { D P } } )$ . The centered summands have norm at most $2 T$ and second moment at most $D _ { p }$ . By Lemma B.1, with probability at least $1 - \beta / 4$

$$
\Vert \widehat { m } - m _ { T } \Vert _ { 2 } \leq C _ { 1 } \left( \sqrt { \frac { D _ { p } \ell } { n } } + \frac { T \ell } { n } \right) .
$$

Gaussian concentration gives, with probability at least $1 - \beta / 4 , \| Z \| _ { 2 } \le \sigma \left( \sqrt { D _ { p } } + \sqrt { 2 \ell } \right)$ . Orthonormality and $| U | \le 1$ imply $\| m _ { T } \| _ { 2 } \leq 1$ . Projection onto the unit ball therefore cannot increase distance to m<sub>T</sub>. Using $T = 2 \sqrt { D _ { p } }$ , with probability at least $1 - \beta / 2$ we obtain

$$
\begin{array} { r l } & { \left\| \widetilde { m } - m _ { T } \right\| _ { 2 } \leq \left\| \widehat { m } - m _ { T } \right\| _ { 2 } + \left\| Z \right\| _ { 2 } } \\ & { \qquad \leq C _ { 2 } \left( \sqrt { \frac { D _ { p } \ell } { n } } + \frac { \sqrt { D _ { p } \ell } } { n } + \frac { \left( D _ { p } + \sqrt { D _ { p } \ell } \right) \sqrt { L _ { \delta _ { \mathrm { D P } } } } } { \varepsilon _ { \mathrm { D P } } n } \right) . } \end{array}
$$

For a suficiently large universal constant $K$ , the condition

$$
n \geq K \left( \frac { D _ { p } \ell } { \alpha ^ { 2 } } + \frac { D _ { p } \ell \sqrt { L _ { \delta _ { \mathrm { D P } } } } } { \varepsilon _ { \mathrm { D P } } \alpha } \right)\tag{31}
$$

makes this at most c<sub>0</sub>cα. By Equation (9),

$$
L _ { s } ( q ) - \operatorname* { i n f } _ { Q \in \mathcal { P } _ { p } } L _ { s } ( Q ) = \| \widetilde { m } - m _ { T } \| _ { 2 } ^ { 2 } \leq c _ { 0 } ^ { 2 } c ^ { 2 } \alpha ^ { 2 } .
$$

Finally, projection and orthonormality give ∥q∥ $L ^ { 2 } ( \mu ) = \| \widetilde { m } \| _ { 2 } \leq 1$ on every run.

Proof of Theorem 3.1. Round n up to an even integer. If one observation in $S _ { 1 }$ changes, the private polynomial release is followed only by post-processing with $S _ { 2 }$ fixed. If one observation in $S _ { 2 }$ changes, the distribution of $q$ is unchanged, and conditional on $q ,$ selection is $\varepsilon _ { \mathrm { D P } ^ { \mathrm { - D P } } }$ . Thus, the complete algorithm is $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$

Since $p = \lceil C _ { 0 } / ( c \alpha ) ^ { 2 } \rceil$ , choosing $C _ { \mathrm { d i m } } \ge 2 ( C _ { 0 } + 1 ) ^ { 2 } c ^ { - 5 }$ ensures $d \geq 2 p ^ { 2 } / ( c \alpha )$ . Thus, the dimension condition needed in Lemma 3.5 follows from the theorem’s assumption. The first two terms in the sample bound imply Equation (31) and hence the event in Lemma 3.4 with probability at least $1 - \beta / 2$ . On this event, Lemma 3.5 gives

$$
\operatorname* { m i n } _ { h \in \mathcal { H } ( q ) } \mathrm { e r r } _ { P } ( h ) \le \mathrm { O P T } + ( C / 2 + 1 ) c \alpha .
$$

Condition on any such released polynomial and write $K = | \mathcal { H } ( q ) |$ . Applying Lemma B.5 to the $n / 2$ observations in $S _ { 2 }$ gives, with probability at least $1 - \beta / 2$

$$
\operatorname { e r r } _ { P } ( \widehat { h } ) \leq \operatorname* { m i n } _ { h \in \mathcal { H } ( q ) } \operatorname { e r r } _ { P } ( h ) + 2 \sqrt { \frac { \log ( 8 K / \beta ) } { n } } + \frac { 4 \log ( 4 K / \beta ) } { \varepsilon _ { \mathrm { D P } } n } .
$$

Since log $K = \mathcal { O } \left( \alpha ^ { - 4 } \log ( 1 / \alpha ) \right)$ , the last two terms in the theorem’s sample bound control the dependence on K. The first two terms also control the remaining confidence terms, since $D _ { p } \geq 1$ With suficiently large constants in the sample bound, the selection error is at most cα.

A union bound therefore gives, with probability at least $1 - \beta _ { ; }$

$$
\mathrm { e r r } _ { P } ( \widehat { h } ) \leq \mathrm { O P T } + ( C / 2 + 2 ) c \alpha \leq \mathrm { O P T } + \alpha ,
$$

where the last inequality follows by taking c suficiently small. Every member of $\mathcal { H } ( q )$ is a halfspace, so the learner is proper.

Finally, we analyse the running time. Computing the clipped Hermite features of the samples in $S _ { 1 }$ takes $\mathcal { O } \left( n p D _ { p } \right)$ time, and the noisy average and its projection take $\mathcal { O } \left( n D _ { p } \right)$ more. Since $\partial _ { i } h _ { \kappa } = \sqrt { \kappa _ { i } } h _ { \kappa - e _ { i } }$ , the matrix $M = \mathbb { E } [ \nabla q ( X ) \nabla q ( X ) ^ { \top } ]$ can be computed exactly from the coeficients of $q$ in $\mathcal { O } \left( d ^ { 2 } D _ { p } \right)$ time, and its eigendecomposition takes $\mathcal { O } \left( d ^ { 3 } \right)$ time. By Lemma 3.2, $| \mathcal { H } ( q ) | =$ exp $\left( \mathcal { O } \left( \alpha ^ { - 4 } \log ( 1 / \alpha ) \right) \right)$ ; each member can be listed in $\mathrm { p o l y } ( \dot { d } , \dot { 1 / } \alpha )$ time and evaluated on $S _ { 2 }$ in $\mathcal { O } \left( n d \right)$ time. Thus, the exponential mechanism takes $\mathcal { O } \left( n d | \mathcal { H } ( q ) | \right)$ time overall. Since $d \leq D _ { p } \leq n$ and $1 / \alpha \leq n$ , the total running time is poly $\cdot ( n ) \cdot \exp \bigl ( \mathcal { O } \left( \alpha ^ { - 4 } \log ( 1 / \alpha ) \right) \bigr )$ □

## F Missing Details for Semi-Agnostic Learning

Algorithm 4 DP Semi-Agnostic Halfspace Learner   
Require: Dataset $D = ( ( x _ { 1 } , y _ { 1 } ) , \dots , ( x _ { n } , y _ { n } ) )$ , privacy parameters $\varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } > 0 .$ accuracy parameter   
α, error probability $\beta .$   
Partition dataset for multiple runs and set parameters.   
1: Set $K  C _ { K } \log ( 1 / \alpha )$ for universal constant $C _ { K } \geq 1$   
2: Let $D _ { v a l } \subset D ,$ , such that $\begin{array} { r } { | D _ { v a l } | = \mathcal { O } \left( \frac { \log ( K / \beta ) } { \alpha ^ { 2 } } + \frac { \log ( K / \beta ) } { \alpha \varepsilon _ { \mathrm { D P } } } \right) } \end{array}$ . Set $D ^ { \prime } \gets D \setminus D _ { v a l } .$   
3: Partition D<sup>′</sup> into K equal parts $D ^ { 1 } , \ldots , D ^ { K }$   
4: Set $\eta _ { 1 }  C _ { \eta } \alpha$ , and $\eta _ { k }  2 ^ { k - 1 } \eta _ { 1 }$ for $k \in [ K ]$ and universal constant $C _ { \eta } > 1 .$   
Run Algorithm 3 on each partition with appropriate parameters.   
5: for $k = 1$ to K do   
6: Set $\begin{array} { r } { h _ { k } : = ( \hat { w } _ { k } , \hat { \tau } _ { k } ) \gets \mathrm { D P S A S } _ { \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } , \eta _ { k } } ( D ^ { k } ) . } \end{array}$   
7: end for   
Run all $h _ { k } \mathbf { ' s }$ on $D _ { v a l }$ and pick the best one using exponential mechanism.   
8: Let $H \gets ( h _ { 1 } , \ldots , h _ { K } )$   
9: Define the score function Score $\textstyle \left( D _ { v a l } , h _ { k } \right) : = { \frac { 1 } { | D _ { v a l } | } } \sum _ { \mathrm { ~ \dots ~ } \cap \mathrm { ~ \Gamma ~ } } \mathbf { 1 } \{ \operatorname { s i g n } ( { \hat { w } } _ { k } \cdot x + { \hat { \tau } } _ { k } ) = y \}$   
(x,y)∈D<sub>val</sub>   
10: Set $\Delta  1 / | D _ { v a l } |$   
11: Set $\bar { h } \gets \mathrm { E x p M e c h } _ { \varepsilon _ { \mathrm { D P } } , H } ( \mathsf { S c o r e } , \Delta , D _ { v a l } ) .$   
12: Return ${ \bar { h } } .$

## F.1 Privacy

We first prove the privacy guarantees of Algorithm 4.

Proposition F.1. Algorithm 4 satisfies $\left( \mathcal { O } \left( \varepsilon _ { \mathrm { D P } } \right) , \mathcal { O } \left( \delta _ { \mathrm { D P } } \right) \right) \cdot \mathrm { D P } .$

Proof. We show that all diferent steps of Algorithm 4 that involve referencing the data using Algorithm 3 and the exponential mechanism satisfy DP, and then we apply parallel composition. However, for that, we first need to prove the privacy guarantees of Algorithm 3 by combination of using privacy guarantees of the standard DP mechanisms and parallel composition.

In Algorithm 3, when computing the threshold τˆ, the sensitivity of $\mathbb { E } _ { ( x , y ) \sim D } [ y ]$ (the average of all the labels) has sensitivity $2 / m _ { 0 }$ because the labels can be either −1 or 1. Therefore, adding Laplace noise scaled to $2 / \varepsilon _ { \mathrm { D P } } m _ { 0 }$ satisfies $\varepsilon _ { \mathrm { D P ^ { - } } } \mathrm { D P }$

For computing the initial weight vector wˆ<sub>0</sub>, we compute the noisy average of xy over the entire dataset $D _ { i n i t }$ . Since we clip all the data points to within L in $\ell _ { 2 }$ norm, and y is either −1 or 1, the $\ell _ { 2 }$ sensitivity of this average is $2 L / m _ { 0 }$ . Thus, adding Gaussian noise to this average scaled to $2 L \sqrt { 2 \log ( 2 / \delta _ { \mathrm { D P } } ) } / \varepsilon _ { \mathrm { D P } } m _ { 0 }$ satisfies $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$

Next, in each iteration of the gradient descent procedure, we add Gaussian noise to the average gradient vector. From Claim 4.3, we see that the average gradient is of the form

$$
\begin{array} { r } { \mathbb { E } _ { ( x , y ) \sim D } \left[ - y \mathbf { 1 } \{ | w \cdot x - b _ { t } | \leq a _ { t } \} \nabla _ { w } \frac { w \cdot x } { \| w \| } \right] . } \end{array}
$$

Therefore, the sensitivity of this average gradient is at most

$$
\frac { 2 \left\| \nabla _ { w } \frac { w \cdot x } { \| w \| } \right\| } { m _ { 0 } } \leq \frac { 2 \| x \| } { m _ { 0 } } \leq \frac { 2 L } { m _ { 0 } } .
$$

This implies that adding Gaussian noise to the average scaled to $2 L \sqrt { 2 \log ( 2 / \delta _ { \mathrm { D P } } ) } / \varepsilon _ { \mathrm { D P } } m _ { 0 }$ satisfies $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ in each iteration. However, in each iteration, a disjoint dataset gets used, so no data point gets used twice for any DP operation. Hence, by parallel composition, we get $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ for this part, as well.

Finally, applying parallel composition over the three steps again, we obtain $( \varepsilon _ { \mathrm { D P } } , \delta _ { \mathrm { D P } } ) – \mathrm { D P }$ for Algorithm 3.

The guarantees of the exponential mechanism in Algorithm 4 imply $\varepsilon _ { \mathrm { D P } ^ { \mathrm { - } } } \mathrm { D P }$ because the sensitivity of the mean of 0-1 random variables is $1 / | D _ { v a l } |$ . Since all the $\mathrm { D P }$ steps in Algorithm 4 are taken on disjoint and separate datasets, we obtain the required DP guarantee using parallel composition.

## F.2 Relevant Assumptions and Regularity Conditions

For simplicity, we assume that the ambient dimension d is large enough. This assumption can be removed by using the exact expressions for concentration inequalities for high-dimensional Gaussians.

Assumption F.2 (Assumption on ambient dimension). We assume the following bound on the ambient dimension in terms of the error probability $\beta \in ( 0 , 1 )$ and number of samples $n \in \mathbb N$ $d \geq 1 6 \log ( n / \beta )$

We do this for ease of notation, but we can make our calculations exact with a few extra lower order terms by not making this assumption.

Next, we state the Laurent-Massart inequality for concentration of high-dimensional Gaussians.

Lemma F.3 (Laurent-Massart inequality). Let $X \sim { \mathcal { N } } ( 0 , \mathbb { I } )$ be d-dimensional, and $\Sigma = A ^ { \top } A$ for $A \in \mathbb { R } ^ { d \times d }$ . Then the following holds for all $t > 0$

$$
\operatorname* { P r } [ \| A X \| _ { 2 } ^ { 2 } > \operatorname { t r } ( \Sigma ) + 2 { \sqrt { \operatorname { t r } ( \Sigma ^ { 2 } ) t } } + 2 t \| \Sigma \| _ { 2 } ] \leq e ^ { - t }
$$

$$
\mathrm { P r } [ \| A X \| _ { 2 } ^ { 2 } < \mathrm { t r } ( \Sigma ) - 2 \sqrt { \mathrm { t r } ( \Sigma ^ { 2 } ) t } ] \le e ^ { - t }
$$

Using the above, we have the following deterministic regularity condition that we would assume throughout the proof.

Lemma F.4 (Concentration of high-dimensional Gaussians). Let $X _ { 1 } , \ldots , X _ { n }$ be i.i.d. samples from a d-dimensional Gaussian $\mathcal { N } ( 0 , \mathbb { I } )$ . Then for $\beta \in ( 0 , 1 )$ , the following holds with probability at least $1 - \beta$

$$
\operatorname* { P r } \left[ \forall ~ i \in [ n ] : \| X _ { i } \| \in [ \sqrt { d } / 2 , 2 \sqrt { d } ] \right] \ge 1 - \beta
$$

Proof. The proof follows by simply applying Assumption F.2, Lemma F.3, and the union bound over all n samples. □

In Algorithm 3, we set the clipping parameter $L = 2 { \sqrt { d } }$ to ensure that all data points lie completely within the $\ell _ { 2 }$ ball of radius L around the origin, and no point gets actually clipped with high probability.

## F.3 Missing Proofs from Section 4

Proof of Claim 4.3. The proof of this claim is the same as that of Claim C.2 from (Diakonikolas et al., 2022), except that the noisy gradient term has Gaussian noise added in the subspace orthogonal to w (which is a known DP quantity) because the weight vectors get ultimately projected onto the unit ball and the GD algorithm will not move in the direction of w. Therefore, by Pythagoras’ theorem, $\boldsymbol { w } - \lambda \nabla _ { \boldsymbol { w } } \tilde { \mathcal { L } } _ { t } ( \boldsymbol { w } )$ term will still have norm at least 1 as the gradient and the noise belong to the same subspace that is orthogonal to w. □

Proof of Claim 4.5. In our algorithm $\tilde { w } _ { 0 } = w _ { 0 } + z _ { 0 }$ , where $\begin{array} { r } { z _ { 0 } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } \mathbb { I } ) , \sigma = \frac { 4 \sqrt { 2 d \log ( 2 / \delta _ { \mathrm { D P } } ) } } { \varepsilon \mathsf { n p } m \mathsf { n } } } \end{array}$ and w is as defined by Diakonikolas et al. (2022) in Algorithm 2. We know from the proof of Theorem 1.2 in (Diakonikolas et al., 2022) that $w ^ { \ast } \cdot w _ { 0 } > \omega ( \eta )$ . Therefore, it is suficient to bound $w ^ { * } \cdot z _ { 0 }$ because $w ^ { * } { \cdot } \tilde { w } _ { 0 } > 0 \implies w ^ { * } { \cdot } \hat { w } _ { 0 } > 0$ . Note that for any fixed unit vector w<sup>∗</sup>, $w ^ { * } { \cdot } z _ { 0 } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ Applying concentration inequality for 1-D Gaussians, we get that with probability at least $1 - \mathcal { O } \left( \beta \right)$ $\begin{array} { r } { \left. w ^ { * } \cdot z _ { 0 } \right. \leq \mathcal { O } \left( \frac { \sqrt { d \log ( 1 / \delta _ { \mathrm { D P } } ) \log ( 1 / \beta ) } } { \varepsilon _ { \mathrm { D P } } m _ { 0 } } \right) < \mathcal { O } \left( \eta \right) } \end{array}$ by plugging in our bound on the sample complexity.