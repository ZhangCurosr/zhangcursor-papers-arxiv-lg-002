# Why not to use the Gaussian kernel

Toni Karvonen

toni.karvonen@lut.fi

School of Engineering Sciences

Lappeenranta–Lahti University of Technology LUT Yliopistonkatu 34, 53850 Lappeenranta, Finland

Chris J. Oates

chris.oates@ncl.ac.uk

School of Mathematics, Statistics & Physics

Newcastle University

Newcastle upon Tyne, NE1 7RU, United Kingdom

## Abstract

Kernels measure similarity or correlation in tasks such as regression and classification. The Gaussian kernel, other names of which include squared exponential and radial basis function kernel, is one of the most popular in Gaussian process regression. We argue that the Gaussian kernel is best avoided and should never be used as a default. The argument rests on two results demonstrating that the Gaussian kernel is extremely brittle. First, the Gaussian kernel gives rise to a conditional variance that is unrealistically small. If the variance is used to quantify predictive uncertainty, catastrophic overconfidence is almost inevitable. Second, a small variance goes hand in hand with numerical ill-conditioning, so that to use the Gaussian kernel in practice requires tricks such as nugget terms that efectively modify the underlying regression or classification model. These problems are caused by the unnatural smoothness of the Gaussian kernel, a fact we are far from the first to take notice of. The problem is not the Gaussian form itself but the analyticity of the kernel: Our argument is more broadly that analytic kernels are best avoided. For stationary kernels analyticity is essentially equivalent to an exponential decay of the spectral density. Keywords: Gaussian kernel, squared exponential kernel, Gaussian process regression, uncertainty quantification

## 1. Introduction

Kernels measure similarity or correlation between inputs and underpin many popular methods for tasks such as regression, classification, principal component analysis, and space-filling design. Although a variety of needs have bred a great variety of kernels, some have always remained more popular than others. The purpose of this article is to make a case against the persistent use of the Gaussian kernel in Gaussian process regression. Let $\ell > 0$ be a lengthscale parameter. In regression, the Gaussian kernel (see Figure 1)

$$
k ( x , y ) = \exp \left( - \frac { \| x - y \| _ { 2 } ^ { 2 } } { 2 \ell ^ { 2 } } \right)\tag{1.1}
$$

measures the correlation of inputs $x , y \in \mathbb { R } ^ { d }$ . Also known as squared exponential, exponentiated quadratic, and radial basis function kernel, the Gaussian is probably the most popular kernel for Gaussian processes and kernel methods in machine learning. Its popularity goes back to the very introduction of Gaussian processes to machine learning (Williams and Rasmussen, 1996; Neal, 1998). The Gaussian kernel is infinitely smooth, which is to say that one can diferentiate it arbitrarily many times. In fact, it is much more than infinitely smooth: It is analytic with infinite radius of convergence, which means that its Taylor series developed at any point converges on the whole of $\mathbb { R } ^ { d }$ . It is well recognised that the Gaussian kernel is popular despite severe flaws it has:

![](images/cdebb730be2efc4038a311a5ef9218236b545229a7ef6f095b73c6d33bc7b210.jpg)  
Figure 1: Three kernels on $\mathbb { R } ^ { 2 }$ that appear in this article: The Gaussian kernel in (1.1), the inverse multiquadric kernel in (4.4), and a Mat´ern kernel in (2.2) with $\nu = 1 / 2$ The plots depict kernel translates at the origin, k(0, ·). All kernels use lengthscale ℓ = 1.

“Stein [1999] argues that such strong smoothness assumptions are unrealistic for modelling many physical processes, and recommends the Mat´ern class [...]. However, the squared exponential is probably the most widely-used kernel within the kernel machines field.” — Rasmussen and Williams (2006, p. 83)

Section 5 collects a number of reasons that we believe have contributed to the enduring popularity of the Gaussian kernel.

## 1.1 Contributions

Even though it has seen three decades of successful usage in machine learning (and much more in other fields), we argue that the Gaussian kernel should not be used in Gaussian process regression. As is evident from the quote above and further citations below, we are far from the first to make this argument. However, no comprehensive case against the Gaussian relying on mathematically rigorous results on its failure in practical scenarios has been made. This article makes such a case. We do not purport to demonstrate that the Gaussian kernel never works; there is ample evidence to show that it often does. Rather, we argue that, in the absence of extremely strong prior information, one should never default to the Gaussian kernel in applications. At the time of writing, this recommendation is violated by many popular software packages, such as GPy (GPy, since 2012) and scikit-learn (Pedregosa et al., 2011). By using the Gaussian kernel one makes, knowingly or not, an analyticity assumption that the function being modelled is completely determined by its values in the neighbourhood of any given point. This assumption is rarely, if ever, true. While this alone might be (and has been) enough to convince some to shy away from the Gaussian kernel, the actual argument we make rests on two conspicuous implications of analyticity.

Problem 1: Overconfident uncertainty quantification. Gaussian processes provide means for prediction at unseen input locations and quantification of predictive uncertainty via conditional mean and variance functions (see Section 2). The magnitude of the variance at a given point informs the user on the accuracy of the conditional mean. If the variance is small at a point $x ,$ the mean is understood to be a good approximation to the underlying latent function that generated the data. Typically it is reasonable to expect the variance at x to be small only if the dataset contains input points that lie close to x. This is not what happens when the Gaussian kernel is used. Figure 2 plots the conditional variance over $[ - 2 , 2 ] ^ { 2 } \subset \mathbb { R } ^ { 2 }$ for three diferent kernels when data are available only on the small sub-domain $\bar { [ 1 , 2 ] ^ { 2 } } \subset \mathbb { R } ^ { 2 }$ . For the finitely smooth Mat´ern- $_ { - } / 2$ and Mat´ern-3/2 kernels [see (2.2)] the variance behaves as expected, increasing rapidly as one moves away from the sub-domain that contains the input points. In contrast, for the Gaussian kernel the variance is much smaller on $[ - 2 , 2 ] ^ { 2 }$ and essentially zero on $[ 0 , 1 ] ^ { 2 }$ , meaning that the model assigns little to no uncertainty for predictions made anywhere on the domain. If the Gaussian kernel is used to quantify uncertainty in predictions when the domain of interest is not suficiently covered by the input points, there is a real risk of overconfident uncertainty quantification that could prove disastrous in downstream tasks or in safety-critical applications. The behaviour of the variance associated to the Gaussian kernel that is summarised in Figure 2 is explained and quantified in Theorems 3.1, 3.2, and 3.5. Estimation of the lengthscale parameter ℓ can provide no remedy: Shrinking ℓ as more data are obtained slows down the rate of decay of the variance uniformly across the domain. If, under a fixed lengthscale, the variance tends to zero everywhere even though all data are concentrated on a small sub-domain, it does so also when the lengthscale is estimated. Remark 3.3 expands upon hyperparameter estimation.

Problem 2: Numerical ill-conditioning. To implement Gaussian process interpolation and many other kernel-based regression and classification methods one must solve a linear system defined by the kernel matrix

$$
{ \sf K } _ { n } = ( k ( x _ { i } , x _ { j } ) ) _ { i , j = 1 } ^ { n } \in \mathbb { R } ^ { n \times n } ,
$$

where $x _ { i }$ are the input points. If k is the Gaussian kernel, solving the linear system is practically impossible unless n is very small. Figure 3 plots the condition number of $\mathsf { K } _ { n }$ as a function of n for four diferent kernels. For the Gaussian kernel the condition number blows up super-exponentially fast, which implies that Gaussian process interpolation cannot be implemented in practice. To combat numerical ill-conditioning one can then introduce a nugget term, a technique that is equivalent to assuming that the data are noisy. For example, GPy defaults to the nugget $\sigma ^ { 2 } = 1 0 ^ { - 8 }$ and GPyTorch (Gardner et al., 2018) to $\sigma ^ { 2 } = 1 0 ^ { - 6 }$ . Opinions difer on whether or not nuggets ought to be used (in particular, see Gramacy and Lee, 2012). Regardless, what seems irrefutable to us is that applying a nugget just to make sure that a more or less arbitrarily chosen model can be implemented stably is not the correct approach to take. That is, if one has little prior information to choose the model with, defaulting to the Gaussian kernel introduces completely unnecessary numerical dificulties that could have been largely avoided by defaulting to a less smooth kernel, such as a Mat´ern. It is counterproductive to use the Gaussian kernel if in doing so one must, by introducing a nugget, modify the model. One could as well start with a numerically more stable kernel. Ill-conditioning results for the Gaussian kernel are given in Theorem 3.6 and Corollary 3.7.

![](images/481edcdf5c0bb78016da72749b13272594d9e0e6b8e14a46005becde3024d2b4.jpg)  
Figure 2: Conditional variance in (2.3) in the interpolatory setting (i.e., $\sigma = 0 )$ for two Mat´ern kernels defined in (2.2) and the Gaussian kernel in (1.1) over the domain $D = [ - 2 , 2 ] ^ { 2 } \subset \mathbb { R } ^ { 2 }$ . All kernels have lengthscale $\ell = 2$ The input points $X _ { n } = \{ x _ { 1 } , \ldots , x _ { n } \}$ (red) form a grid on the sub-domain $[ 1 , 2 ] ^ { 2 } \subset [ - 2 , 2 ] ^ { 2 }$ . The Gaussian kernel is covered by Theorem 3.2; how Mat´erns behave is explained by Theorem 4.1.

The Gaussian kernel is not the only kernel subject to such problems. Section 4 takes a more general approach and identifies classes of stationary kernels that sufer from Problems 1 and 2. In short, a stationary kernel whose spectral density decays at least exponentially is analytic and sufers from these problems (Theorem 4.3). The spectral density of the Gaussian kernel is proportional to $\mathrm { e x p } ( - c _ { 1 } \| \omega \| _ { 2 } ^ { 2 } )$ for certain $c _ { 1 } > 0 ;$ the inverse quadratic kernel whose behaviour mimics that of the Gaussian in Figure 3 has spectral density proportional to $\exp ( - c _ { 2 } | \omega | )$ for $c _ { 2 } > 0$ if $d = 1$ . The problems are avoided if the spectral density decays sub-exponentially (Theorems 4.1 and 4.4). Mat´erns, which have little in common with the Gaussian in Figures 2 and 3, have polynomially decaying spectral densities. Kernels that we call Gevrey kernels have sub-exponential spectral densities of the form $\mathrm { e x p } ( - \rho \| \omega \| _ { 2 } ^ { \eta } )$ for $\rho > 0$ and $\eta \in ( 0 , 1 )$ . These kernels are notable in being infinitely diferentiable but non-analytic. Few Gevrey kernels are available in closed form. The most important exception is the case $\eta = 2 / 3$ and $d = 1$ , which yields a kernel that can be expressed in terms of a Whittaker function. To the best of our knowledge, Gevrey kernels have not previously appeared in the literature on Gaussian processes.

In this article we prefer readability over optimality and have strived to provide a rather self-contained account. While many results we present could be slightly improved by invoking existing results in the literature, in no case would this alter their interpretation. For example, it is straightforward to use results in Yarotsky (2013a) and Karvonen and Suzuki (2026) to optimise the exponential term in Theorems 3.1 and 3.2 and Corollary 3.7. Karvonen (2026)

![](images/6ccaf90674375480342cc8e6f07724f2f26dec9d0b4381ad5f8c9cee323676f0.jpg)  
Figure 3: Condition number [see (3.2)] of the kernel matrix ${ \sf K } _ { n } = ( k ( x _ { i } , x _ { j } ) ) _ { i , j = 1 } ^ { n } \in \mathbb { R } ^ { n \times n }$ for $\mathbf { M a t \acute { e } r n – 1 / 2 }$ , Mat´ern-3/2 [see (2.2)], inverse quadratic [see $\left( 4 . 5 \right) ]$ , and Gaussian [see (1.1)] kernels on R. All kernels have lengthscale $\ell = 2$ . For each n the input points are equispaced on the unit interval. Computations were done in doubleprecision arithmetic. Results for the Gaussian and inverse quadratic kernels are unreliable after $n \approx 1 0$ due to limited precision. The largest condition number that can be resolved in double-precision arithmetic is somewhere between $1 0 ^ { 1 6 }$ and $1 0 ^ { 2 0 }$ . For $n \geq 1 0$ the actual condition numbers for the Gaussian and inverse quadratic kernels are significantly larger than shown in the figure. For example, a lower bound on the condition number of $\mathsf { K } _ { n }$ for $n = 2 1$ and the Gaussian kernel given by Corollary 3.7 is approximately $1 . 2 8 \times 1 0 ^ { 2 4 }$

gives an account that is similar in flavour to Section 4 but based on much more advanced tools from harmonic analysis.

## 1.2 Case against the Gaussian in the literature

We are not the first to take note of Problems 1 and 2. Although Stein (1999, pp. 30, 55 and 69–70) has laid out a strong case against the use of the Gaussian kernel in modelling due to its infinite smoothness, his arguments need not be persuasive to all practitioners, being either empirical (pp. 69–70) or applying to a setting that, as he freely admits, rarely occurs in practice (p. 30): “That is, it is possible to predict $Z ( t )$ perfectly for all $t > 0$ based on observing $Z ( s )$ for all $s \in ( - \epsilon , 0 ]$ for any $\epsilon > 0 . ^ { \mathfrak { n } }$ See also Handcock and Stein (1993, p. 406). Such results on “predicting the future based on the past” have a long history and go back at least to the works of Kolmogorov (1941) and Wiener (1949, p. 54).<sup>1</sup> A more recent tradition is to be found in kriging and optimisation literature, where the Gaussian kernel has been proved to lack the no-empty-ball property (Vazquez and Bect, 2010b,a; Yarotsky, 2013a; Petit et al., 2022). A kernel is said to have this property if the convergence to zero of the conditional variance as the number of data increases is equivalent to the denseness of the sequence of sampling locations in the domain of interest. In particular, variants of some of our main results could be derived directly from Theorem 2 in Yarotsky (2013a). The screening efect is a related phenomenon discussed in geostatistical literature. The screening efect is said to hold for a given stochastic process if the prediction at a point x depends “mostly” only on observations close to x. Stein (2002, 2011, 2015) has made this notion precise and shown that the screening efect does not hold if the kernel of a Gaussian process is Gaussian or inverse quadratic.

One attempting to invert the kernel matrix $\mathsf { K } _ { n } = ( k ( x _ { i } , x _ { j } ) ) _ { i , j = 1 } ^ { n }$ when k is Gaussian is bound to observe its extreme ill-conditioning. This observation has been made repeatedly in the literature. For example, see Example 1 in Diamond and Armstrong (1984), Section 4.3 in Lewis (1987), the work of Posa (1989) whose conclusions<sup>2</sup> are much in the spirit of this article, pages 101–2 in Ababou et al. (1994), Section 2 in Peng and Wu (2014), page 3063 in Gu et al. (2018), and Belkin (2018). How condition numbers of kernel matrices behave is relatively well understood in the literature on radial basis functions (Schaback, 1995; Diederichs and Iske, 2019). See Chapter 12 in Wendland (2005) for a review. The general picture is that the smoother the kernel, the larger the condition number (Schaback and Wendland, 2006, Guideline 3.13). It is interesting to note that, except when it comes to numerical ill-conditioning, few remarks on the disadvantages of the Gaussian kernel seem to exist in the radial basis function literature, which serves as a foundation for most convergence theory of Gaussian process interpolation. Convergence results are almost invariably expressed in terms of the fill-distance $h = \mathrm { s u p } _ { x \in D } \operatorname* { m i n } _ { i = 1 , \ldots , n } \| x - x _ { i } \| _ { 2 }$ of the input points $x _ { 1 } , \ldots , x _ { n }$ in a bounded domain of interest $D \subset \mathbb { R } ^ { d }$ The fill-distance tends to zero if the sequence of input points, $( x _ { i } ) _ { i = 1 } ^ { \infty } ,$ is dense in D. Standard references include Wendland (2005, Section 11.4) and Rieger and Zwicknagl (2010). Yarotsky (2013b) is one of the few exceptions known to us that does not use the fill-distance. We also remark that Platte (2011) has established interesting connections between approximation by polynomials and Gaussians.

## 1.3 What then?

To attempt to give a general recommendation of which kernel to use is largely futile. But let us assume that one has resolved to use a stationary kernel. Whether or not that is a reasonable thing to do is not a debate that we wish to enter into in this article.<sup>3</sup> Our main message is that the Gaussian kernel should be avoided. Likewise, analytic kernels (i.e., the spectral density decays exponentially or faster) are to be avoided. We still find the old recommendation by Stein (1999) to use a finitely smooth Mat´ern a sensible course of action and consider a low-order Mat´ern, such as one of order $\nu = 3 / 2$ or $\nu = 5 / 2$ , a reasonable default in the absence of real prior information. If one is convinced that the latent function is infinitely diferentiable but wants to steer clear of the Gaussian kernel (as one should), inverse multiquadrics in (4.4), which are “barely” analytic, and Gevrey kernels in Section 4.3 are noteworthy options. As closed form expressions for Gevrey kernels are unavailable when $d > 1$ , one would have to resort to product kernels. However, this need not be a problem even if one were averse to product kernels, for smooth product kernels tend to look very much like their isotropic cousins (see Figure 4).

![](images/9801eb2fb22c403399fdeb029dbb6943c7eac3852357c1be7620c5f024b76c28.jpg)  
Figure 4: Isotropic and product forms on $\mathbb { R } ^ { 2 }$ of the $\mathbf { M a t e r n { - } 3 / 2 }$ kernel in (2.2) and the inverse multiquadric kernel in (4.4). The plots depict kernel translates at the origin, $k ( 0 , \cdot )$ Isotropic kernels have the form $k ( x , y ) = \phi ( \| x - y \| _ { 2 } )$ for $\phi \colon [ 0 , \infty ) $ R and product kernels the form $k ( x , y ) = \phi ( | x _ { 1 } - y _ { 1 } | ) \cdot \cdot \cdot \phi ( | x _ { d } - y _ { d } | )$ , where $x = ( x _ { 1 } , \ldots , x _ { d } ) \in \mathbb { R } ^ { d }$ and $y = ( y _ { 1 } , \dots , y _ { d } ) \in \mathbb { R } ^ { d }$ . All kernels use lengthscale $\ell = 1$

## 2. Gaussian processes

Rasmussen and Williams (2006) provide a now-standard treatment of Gaussian processes for machine learning. Other useful references include Chapter 2 in Berlinet and Thomas-Agnan (2004); Gramacy (2020); and Kanagawa et al. (2025).

## 2.1 Kernels and priors

Let m : $\mathbb { R } ^ { d } $ R be a function and $k \colon  { \mathbb { R } } ^ { d } \times  { \mathbb { R } } ^ { d } \to  { \mathbb { R } }$ a symmetric and strictly positive-definite kernel, which is to say that

$$
\sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } a _ { i } a _ { j } k ( x _ { i } , x _ { j } ) > 0\tag{2.1}
$$

for any $n \geq 1$ , any pairwise distinct points $x _ { 1 } , \ldots , x _ { n } \in \mathbb { R } ^ { d }$ , and any non-zero vector $( a _ { 1 } , \ldots , a _ { n } ) \in \mathbb { R } ^ { n }$ . The Gaussian kernel

$$
k ( x , y ) = \exp \left( - \frac { \| x - y \| _ { 2 } ^ { 2 } } { 2 \ell ^ { 2 } } \right)
$$

with lengthscale $\ell > 0$ is a basic example of a strictly positive-definite kernel, which follows from Bochner’s theorem (Wendland, 2005, Sec. 6.2) and the fact that the Fourier transform of a Gaussian function is Gaussian. Mat´ern kernels constitute another important class of strictly positive-definite kernels. Let $\nu > 0$ be a smoothness parameter and $\kappa _ { \nu }$ the modified Bessel function of the second kind. The Mat´ern kernel of smoothness ν is given by

$$
k _ { \nu } ( x , y ) = \frac { 2 ^ { 1 - \nu } } { \Gamma ( \nu ) } \left( \frac { \sqrt { 2 \nu } \left\| x - y \right\| _ { 2 } } { \ell } \right) ^ { \nu } \mathcal { K } _ { \nu } \left( \frac { \sqrt { 2 \nu } \left\| x - y \right\| _ { 2 } } { \ell } \right) .\tag{2.2}
$$

Every Mat´ern kernel of half-integer order $( \mathrm { i . e . , } \nu = s + 1 / 2$ for $s \in \mathbb { N } )$ can be written as a product of a polynomial and an exponential. For example,

$$
k _ { \nu = 1 / 2 } ( x , y ) = \exp { \left( - \frac { r } { \ell } \right) } \quad \mathrm { a n d } \quad k _ { \nu = 3 / 2 } ( x , y ) = \left( 1 + \frac { \sqrt { 3 } r } { \ell } \right) \exp { \left( - \frac { \sqrt { 3 } r } { \ell } \right) } ,
$$

![](images/9eaea77b38129d9880592bc9175c8bdb92df5b89889d870b88e2d6c66cf939a7.jpg)  
Figure 5: Samples from Gaussian processes on $\mathbb { R } ^ { 2 }$ with three diferent kernels. Left: Mat´ern kernel with $\nu = 1 / 2$ (continuous but non-diferentiable samples). Middle: Mat´ern kernel with $\nu = 3 / 2$ (once diferentiable samples). Right: Gaussian kernel (infinitely diferentiable samples).

where $r = \| x - y \| _ { 2 }$ . Mat´erns converge pointwise to the Gaussian as $\nu \to \infty$ . We return to this phenomenon in Section 5.4.

A Gaussian process $\mathsf { f } _ { \mathrm { G P } } \sim \mathrm { G P } ( m , k )$ on $\mathbb { R } ^ { d }$ with mean m and covariance kernel k is a stochastic process whose finite-dimensional distributions are multivariate Gaussians with mean and covariance determined by m and k. That is, for any $n \geq 1$ and any collection of points $X _ { n } = \{ x _ { 1 } , \ldots , x _ { n } \} \subset \mathbb { R } ^ { d }$ 2

$$
\left[ \begin{array} { c } { \mathsf { f } _ { \mathbb { G P } } ( x _ { 1 } ) } \\ { \vdots } \\ { \mathsf { f } _ { \mathbb { G P } } ( x _ { n } ) } \end{array} \right] \sim \mathrm { N } \left( \left[ \begin{array} { c } { m ( x _ { 1 } ) } \\ { \vdots } \\ { m ( x _ { n } ) } \end{array} \right] , \left[ \begin{array} { c c c } { k ( x _ { 1 } , x _ { 1 } ) } & { \cdots } & { k ( x _ { 1 } , x _ { n } ) } \\ { \vdots } & { \ddots } & { \vdots } \\ { k ( x _ { n } , x _ { 1 } ) } & { \cdots } & { k ( x _ { n } , x _ { n } ) } \end{array} \right] \right) = : \mathrm { N } ( \mathsf { m } _ { n } , \mathsf { K } _ { n } ) .
$$

The covariance matrix $\mathsf { K } _ { n }$ , henceforth called kernel matrix, is strictly positive-definite and thus invertible whenever the points in $X _ { n }$ are pairwise distinct. In particular,

$$
\mathbb { E } [ \mathsf { f } _ { \mathrm { G P } } ( x ) ] = m ( x ) \quad \mathrm { ~ a n d ~ } \quad \mathrm { C o v } [ \mathsf { f } _ { \mathrm { G P } } ( x ) , \mathsf { f } _ { \mathrm { G P } } ( y ) ] = k ( x , y )
$$

for all $x , y \in \mathbb { R } ^ { d }$ . The covariance kernel determines what the samples from the Gaussian process look like, or how they fluctuate. In particular, the smoothness (i.e., the degree of diferentiability) of the kernel is inherited by the samples. This is illustrated in Figure 5.

## 2.2 Gaussian process regression and interpolation

Let D be a non-empty subset of $\mathbb { R } ^ { d }$ and $f _ { 0 } \colon D  \mathbb { R }$ an unknown latent function. In Gaussian process regression, the unknown function $f _ { 0 }$ is modelled as a Gaussian process $\mathsf { f } _ { \mathrm { G P } } \sim \mathrm { G P } ( m , k )$ that is then conditioned on a set of training data $\mathcal { D } _ { n } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ which consists of potentially noisy observations $y _ { i }$ of $f _ { 0 }$ at a collection of n input points, $X _ { n } = \{ x _ { 1 } , \ldots , x _ { n } \} \subset D$ . We always tacitly assume that the input points are pairwise distinct. It is typical to assume that the observations are corrupted by Gaussian noise with standard deviation $\sigma \geq 0 \colon$

$$
y _ { i } = f _ { 0 } ( x _ { i } ) + \varepsilon _ { i } , \quad \mathrm { ~ w h e r e ~ } \quad \varepsilon _ { i } \sim \mathrm { N } ( 0 , \sigma ^ { 2 } ) \mathrm { ~ a r e ~ i . i . d } .
$$

The conditional process, $\mathbf { f } _ { \mathrm { G P } } \mid \mathcal { D } _ { n } ,$ , is a Gaussian process. Its mean and variance are

$$
\mu _ { \sigma } ( x \mid \mathcal { D } _ { n } ) = \mathbb { E } [ { \mathfrak { f } } _ { \mathrm { G P } } ( x ) \mid \mathcal { D } _ { n } ] = m ( x ) + \mathsf { k } _ { n } ( x ) ^ { \top } ( \mathsf { K } _ { n } + \sigma ^ { 2 } \mathsf { l } _ { n } ) ^ { - 1 } ( \mathsf { y } _ { n } - \mathsf { m } _ { n } )
$$

![](images/f4e07e3bbe6e2919e678c7b9c4da33279df2b08a40c41197b179c6b8ee881465.jpg)  
Figure 6: The conditional mean (black) and the pointwise 95% credible intervals (shaded) obtained from the conditional variance over the interval [0, 1] for the Mat´ern-3/2 and Gaussian kernels with $\ell = 0 . 3$ . The data consist of four observations that are indicated in red. Observe that the mean interpolates the data when $\sigma = 0$

and

$$
\mathbb { V } _ { \sigma } ( x \mid \mathcal { D } _ { n } ) = \mathrm { V a r } [ \mathsf { f } _ { \mathrm { G P } } ( x ) \mid \mathcal { D } _ { n } ] = k ( x , x ) - \mathsf { k } _ { n } ( x ) ^ { \top } ( \mathsf { K } _ { n } + \sigma ^ { 2 } \mathsf { l } _ { n } ) ^ { - 1 } \mathsf { k } _ { n } ( x ) ,\tag{2.3}
$$

where $\mathbf { k } _ { n } ( x ) = ( k ( x , x _ { 1 } ) , \ldots , k ( x , x _ { n } ) )$ and $\mathsf { y } _ { n } = ( y _ { 1 } , \ldots , y _ { n } )$ are column vectors and $\mathsf { I } _ { n }$ the $n \times n$ identity matrix. If $\sigma = 0$ , the latent function is observed without noise, so that $y _ { i } = f _ { 0 } ( x _ { i } )$ . In this case we use the more concise notation

$$
\mu ( x \mid { \mathcal { D } } _ { n } ) = m ( x ) + \mathbf { k } _ { n } ( x ) ^ { \top } \mathbf { K } _ { n } ^ { - 1 } ( \mathbf { y } _ { n } - \mathbf { m } _ { n } ) { \mathrm { a n d } } { \mathbb { V } } ( x \mid { \mathcal { D } } _ { n } ) = k ( x , x ) - \mathbf { k } _ { n } ( x ) ^ { \top } \mathbf { K } _ { n } ^ { - 1 } \mathbf { k } _ { n } ( x ) .\tag{2.4}
$$

To distinguish the cases $\sigma > 0$ and $\sigma = 0$ , we speak of Gaussian process interpolation when $\sigma = 0$ . Most of our arguments against the Gaussian kernel apply to interpolation. The conditional mean and variance for both $\sigma = 0$ and $\sigma > 0$ are depicted in Figure 6. Observe that for $\sigma = 0$ the conditional mean interpolates the data, in that $\mu ( x _ { i } \mid \mathcal D _ { n } ) = f _ { 0 } ( x _ { i } )$ and $\mathbb { V } ( x _ { i } \mid \mathcal { D } _ { n } ) = 0$ for each $i \in \{ 1 , \ldots , n \}$ . The conditional mean provides point estimates for the unobserved outputs $f _ { 0 } ( x )$ while the conditional variance quantifies uncertainty in these estimates via, for example, credible intervals as in Figure 6. Note that the conditional variance in (2.3) and (2.4) does not depend on the observations $y _ { i }$ . The gist of our case against the Gaussian kernel is that it gives rise to a conditional variance that is far too small, a phenomenon that is to some extent observable in Figure 6.

## 3. Why not to use the Gaussian kernel

The gist of our case against the Gaussian kernel

$$
k ( x , y ) = \exp { \left( - \frac { \| x - y \| _ { 2 } ^ { 2 } } { 2 \ell ^ { 2 } } \right) } ,\tag{3.1}
$$

where $x , y \in \mathbb { R } ^ { d } .$ is that it is by far the brittlest among all commonly used kernels for Gaussian process regression in machine learning and spatial statistics. The brittleness is manifested via (a) overconfident uncertainty quantification and (b) numerical ill-conditioning. These issues are discussed in Section 3.1 and Section 3.2, respectively.

## 3.1 Overconfident uncertainty quantification

A Gaussian process model is overconfident if the conditional standard deviation, $\mathbb { V } _ { \sigma } ( \cdot \mid \mathcal { D } _ { n } ) ^ { 1 / 2 }$ is “much” smaller than the true prediction error, $| f _ { 0 } - \mu _ { \sigma } ( \cdot \mid \mathcal { D } _ { n } ) |$ , that it purports to quantify. We refer to Karvonen et al. (2020) for a more thorough discussion on overconfidence. Typically it is unreasonable to expect that a latent function can be predicted well over the entire domain D if observations are available only on a small sub-domain. The Gaussian kernel does not behave as this basic intuition suggests. Recall from Section 2.2 that $\sigma \geq 0$ is the standard deviation of the observation noise.

Theorem 3.1 (Overconfidence for $d = 1 )$ Let $d = 1$ and $\sigma = 0$ . Consider the Gaussian kernel in (3.1) and $D = [ a , b ]$ for $a < b$ . If $X _ { n } = \{ x _ { 1 } , \ldots , x _ { n } \} \subset D$ are any points, then

$$
\operatorname* { s u p } _ { x \in D } \mathbb { V } ( x \mid \mathcal { D } _ { n } ) \leq 2 \bigg ( \frac { \sqrt { 2 } ( b - a ) } { \ell } \bigg ) ^ { 2 n } \frac { 1 } { n ! } .
$$

Proof See Section 7.2. The proof is based on the classical estimate for the error of polynomial interpolation and the connection between the conditional variance and the pointwise worst-case error in the reproducing kernel Hilbert space of k.

Theorem 3.2 (Overconfidence for $d \geq 1 )$ Let $d \geq 1$ and $\sigma = 0$ . Consider the Gaussian kernel in (3.1) and a bounded $D \subset \mathbb { R } ^ { d }$ . If the set $X _ { n } = \{ x _ { 1 } , \ldots , x _ { n } \} \subset D$ contains a Cartesian product $X _ { 1 } ^ { \prime } \times \cdots \times X _ { d } ^ { \prime }$ of d sets of m pairwise distinct reals, then

$$
\operatorname* { s u p } _ { x \in D } \mathbb { V } ( x \mid \mathcal { D } _ { n } ) \leq 2 d \bigg ( \frac { \sqrt { 2 } ( b - a ) } { \ell } \bigg ) ^ { 2 m } \frac { 1 } { m ! } ,
$$

where $a < b$ are such that $D \subset [ a , b ] ^ { d }$ . In particular, if $X _ { n } = X _ { 1 } ^ { \prime } \times \cdots \times X _ { d } ^ { \prime }$ , then

$$
\operatorname* { s u p } _ { x \in D } \mathbb { V } ( x \mid \mathcal { D } _ { n } ) \leq 2 d \left( \frac { \sqrt { 2 } ( b - a ) } { \ell } \right) ^ { 2 n ^ { 1 / d } } \frac { 1 } { ( n ^ { 1 / d } ) ! } .
$$

Proof See Section 7.2. The proof consists of a standard tensor product argument combined with Theorem 3.1.

Theorems 3.1 and 3.2 state that the conditional variance tends to zero with a factorial rate on the entire domain D essentially regardless of where the input points are placed. These theorems are variants of the results discussed in Section 1.2 on predicting the future of a process from past only (Kolmogorov, 1941) and the no-empty-ball property (Vazquez and Bect, 2010b). The conclusion is stronger if $d = 1$ , as then the conditional variance tends to zero for any sequence of sampling points. For $d > 1$ the “bad” point configurations are not artificial as it sufices that $X _ { n }$ contains a d-fold Cartesian product (see Figure 7).<sup>4</sup>

![](images/51d06e1122088edf7824bc61cacef6262df3b6ab383d004668653762050c55b7.jpg)  
Figure 7: Three input point sets $X _ { n }$ on $D = [ - 1 , 1 ] ^ { 2 } \subset \mathbb { R } ^ { 2 }$ that Theorem 3.2 applies to. The Cartesian grids $X _ { 1 } ^ { \prime } \times X _ { 2 } ^ { \prime } \subset X _ { n }$ contained in the sets are emphasised. A two-dimensional Cartesian grid consists of $m \times m$ points.

Overconfidence of uncertainty quantification with the Gaussian kernel was illustrated already in Figure $^ { 2 , }$ which showed that the Gaussian kernel produces a much smaller variance than Mat´erns. Figure 8 shows how the conditional variance tends to zero everywhere on the domain $D = [ - 2 , 2 ] ^ { 2 }$ as more data are obtained even though all sampling points are contained in the small sub-domain $D _ { 0 } = [ 1 , 2 ] ^ { 2 }$ . In a Gaussian process model based on the Gaussian kernel local information is therefore global information: the model thinks that it can learn the latent function globally from local information alone and expresses this belief by shrinking the conditional variance everywhere. This behaviour is akin to (and, as we shall discuss in Section 4.2, a direct consequence of) the fact that an analytic function $f \colon  { \mathbb { R } ^ { d } } \to$ R with infinite radius of convergence can be recovered at any $\boldsymbol { x } \in \mathbb { R } ^ { d }$ from the values of its derivatives at a single point $\boldsymbol { x } _ { 0 } \in \mathbb { R } ^ { d }$ via its Taylor expansion

$$
f ( \boldsymbol { x } ) = \sum _ { \boldsymbol { \alpha } \in \mathbb { N } _ { 0 } ^ { d } } \frac { \mathrm { D } ^ { \boldsymbol { \alpha } } f ( \boldsymbol { x } _ { 0 } ) } { \boldsymbol { \alpha } ! } ( \boldsymbol { x } - \boldsymbol { x } _ { 0 } ) ^ { \boldsymbol { \alpha } } .
$$

None of this needs to be a problem if the user truly believes that their latent function can be learned from local information. But more often than not the Gaussian kernel is used simply as a convenient default. In such a situation one runs the serious risk of unintentional overconfident uncertainty quantification if, for one reason or another, sampling leaves large holes in which the conditional variance is nevertheless very small. Even when the sampling points are space-filling, the conditional variance is likely to decay much faster than is desirable. While Mat´erns and other finitely smooth kernels can induce overconfidence, it is much less severe than under the Gaussian kernel. Moreover, data-driven estimation of hyperparameters for Mat´erns and related kernels often sufices to fix the problem (Szab´o et al., 2015; Karvonen et al., 2020; Naslidnyk et al., 2025; Karvonen and Bachoc, 2025).

Remark 3.3 Hyperparameter estimation cannot fix the global decay of the variance, the fundamental problem associated to the Gaussian kernel that was demonstrated by Theorems 3.1 and 3.2. The problem cannot be circumvented by estimating ℓ (or a multiplicative scaling parameter in front of the kernel) from the data. For selecting a lengthscale $\ell _ { n } ,$ data-driven or not, such that the variance decays with a polynomial rate $n ^ { - 2 s }$ for some $s > 0 ,$ or any other rate more modest than factorial, in the sub-domain that contains the inputs $( e . g . , [ 1 , 2 ] ^ { 2 }$ in Figure 8) simply forces the variance to decay with the same rate everywhere on the domain. For example, setting $\ell _ { n } = \sqrt { 2 e } ( b - a ) n ^ { - 1 / 2 + s / n - 1 / ( 4 n ) }$ in Theorem 3.1 and using Stirling’s approximation for the factorial yields

![](images/fcce7d71794e0700b822d61e26000e5aeb801b01f79c1abdc2fe355bca01d10d.jpg)  
Figure 8: The conditional variance over $[ - 2 , 2 ] ^ { 2 } \subset \mathbb { R } ^ { 2 }$ for the Gaussian kernel in (3.1) with $\ell = 2$ when the input points are located in the sub-domain $[ 1 , 2 ] ^ { 2 }$ . As the number of points increases, the variance decreases rapidly over the entire domain even though the points cover only the small sub-domain.

$$
\operatorname* { s u p } _ { x \in D } \mathbb { V } ( x \mid { \mathcal { D } } _ { n } ) = O ( n ^ { - 2 s } ) .
$$

This is the contraction rate associated to a Mat´ern kernel of order $\nu = s$ . However, for Mat´erns this rate is obtained only when the input sequence is dense in D (see Theorem $4 . 1 ) ;$ for the Gaussian kernel with estimated ℓ there is no such requirement. One could, of course, use a spatially varying lengthscale (or scaling parameter), but this brings about a host of complications common to such non-stationary models. We return to hyperparameter estimation and more general kernel learning in Section 6.

Theorem 3.2 requires that the set of sampling points contain a tensor grid. The proof is based on standard tensor product arguments for transferring one-dimensional bounds to higher dimensions. The conclusion that the variance tends to zero remains true whenever the sequence of points is dense in some open subset of D (see Theorem 3.5). However, in that case we cannot deduce any rate for $d > 1$ . Moreover, the following proposition shows that there are point configurations for which the variance remains bounded from below when $d > 1$

Proposition 3.4 Let $d \ge 1$ and $\sigma \geq 0$ . Consider the Gaussian kernel in (3.1). Let $R _ { 0 } \subset \mathbb { R } ^ { d }$ be the set of roots of a non-trivial d-variate polynomial. If $X _ { n } = \{ x _ { 1 } , \ldots , x _ { n } \}$ for any sequence $\{ x _ { i } \} _ { i = 1 } ^ { \infty } \subset R _ { 0 }$ of pairwise distinct points, then for every $x \notin R _ { 0 }$ there is $c _ { x } > 0$ such that

$$
\mathbb { V } _ { \sigma } ( x \mid \mathcal { D } _ { n } ) \ge c _ { x }
$$

for every $n \in \mathbb { N } .$

![](images/a5ecc96393b6db208bc4dfe3ed9447873d11e4b5905c2c1d3e9f85c1dddba11b.jpg)  
Figure 9: The three bivariate polynomials $- ( x ^ { 2 } + y ^ { 2 } - 1 ) , \ - ( x ^ { 2 } + y ^ { 2 } - 1 ) ( x - y )$ , and $- ( x ^ { 3 } + y ^ { 2 } - 1 )$ along with their roots (in red). Observe that in each case the set of roots is infinite.

Proof See Section 7.2.

When $d = 1$ , the set $R _ { 0 }$ of roots of a non-trivial polynomial is finite. However, when $d > 1$ , the set of roots is in most cases infinite. Easy examples of infinite $R _ { 0 }$ are hyperplanes and the unit sphere $\{ x \in \mathbb { R } ^ { d } : \| x \| _ { 2 } = 1 \}$ , which is the set of roots of the polynomial $1 - ( x _ { 1 } ^ { 2 } + \cdot \cdot \cdot + x _ { d } ^ { 2 } )$ of total degree 2. More examples are given in Figure 9.

Theorems 3.1 and 3.2 show that the variance tends to zero very fast in the interpolatory setting $( \mathrm { i . e . , } \sigma = 0 )$ . The following theorem applies to both interpolation and regression $( \sigma > 0 )$

Theorem 3.5 (Regression for $d \geq 1 )$ Consider the Gaussian kernel in (3.1) for $d \geq 1$ and $\sigma \geq 0 . \ I f X _ { n } = \{ x _ { 1 } , \ldots , x _ { n } \}$ for any sequence $\{ x _ { i } \} _ { i = 1 } ^ { \infty } \subset \mathbb { R } ^ { d }$ of pairwise distinct points whose closure has non-empty interior, then

$$
\operatorname* { s u p } _ { x \in K } \mathbb { V } _ { \sigma } ( x \mid \mathcal { D } _ { n } )  0 \quad \mathrm { ~ } a s \quad n  \infty \quad f o r \ a n y \ c o m p a c t \quad K \ \subset \mathbb { R } ^ { d } .
$$

Proof This is a special case of Theorem 4.3.

There are some noteworthy diferences between Theorems 3.1 and 3.2 and Theorem 3.5. Theorem 3.5 provides no rate of convergence even in the interpolatory case $( \sigma = 0 )$ in which the former theorems imply super-exponential rates. However, unlike Theorem 3.2, which requires that the set of input points contain a grid, Theorem 3.5 makes minimal assumptions about the inputs, only requiring that they be dense in some set.

## 3.2 Numerical ill-conditioning

Our second point is that the Gaussian kernel is numerically brittle. The computation of posterior distributions in Gaussian process regression inevitably involves the kernel covariance matrix $\mathsf { K } _ { n } = ( k ( x _ { i } , x _ { j } ) ) _ { i , j = 1 } ^ { n }$ (see Section 2.2). As we saw in Section 1.2, it has been repeatedly observed that to use the Gaussian kernel can be challenging due to exceptionally poor conditioning of the covariance matrix. If it is assumed that the observations are corrupted by zero-mean Gaussian noise with variance $\sigma ^ { 2 } .$ , the matrix that one needs to invert to compute the conditional mean and variance is ${ \sf K } _ { n } + \sigma ^ { 2 } { \sf I } _ { n }$ , where $\mathsf { I } _ { n }$ is the identity matrix. In this case numerical stability is a non-issue unless σ is very small because the condition number of ${ \sf K } _ { n } + \sigma ^ { 2 } { \sf I } _ { n }$ is bounded from above by $( n + \sigma ^ { 2 } ) / \sigma ^ { 2 }$

However, in tasks such as emulation of computer experiments (Sacks et al., 1989) or probabilistic numerical computation (Hennig et al., 2022) it is natural to assume that the data are noiseless $( \mathrm { i . e . , } \sigma = 0 )$ . Recall that the condition number of a positive-definite matrix A is

$$
\mathrm { c o n d } ( \mathsf { A } ) = \frac { \lambda _ { \operatorname* { m a x } } ( \mathsf { A } ) } { \lambda _ { \operatorname* { m i n } } ( \mathsf { A } ) } ,\tag{3.2}
$$

where $\lambda _ { \operatorname* { m a x } } ( \mathsf { A } )$ and $\lambda _ { \operatorname* { m i n } } ( \mathsf { A } )$ are the largest and smallest eigenvalues of A. The magnitude of the condition number of a covariance matrix is connected to the magnitude of the conditional variance.

Theorem 3.6 Consider a strictly positive-definite stationary kernel $k ( x , y ) = \Phi ( x - y )$ . If $\sigma = 0$ and $X _ { n + 1 } = \{ x _ { 1 } , \dots , x _ { n + 1 } \} \subset \mathbb { R } ^ { d }$ , then

$$
\mathrm { c o n d } ( \mathsf { K } _ { n + 1 } ) \ge \frac { \Phi ( 0 ) } { \mathbb { V } ( x _ { n + 1 } \mid \mathcal { D } _ { n } ) } .
$$

Proof Estimates in Section 12.1 of Wendland (2005) give $\lambda _ { \operatorname* { m i n } } ( \mathsf { K } _ { n + 1 } ) \ \leq \ \mathbb { V } ( x _ { n + 1 } \ | \ \mathcal { D } _ { n } )$ Because the trace is the sum of eigenvalues and tr $( \mathsf { K } _ { n + 1 } ) = ( n + 1 ) \Phi ( 0 )$ by stationarity, not all eigenvalues of $\mathsf { K } _ { n + 1 }$ can be less than $\Phi ( 0 )$

By plugging in the estimates from Theorems 3.1 and 3.2 we obtain the following result on the condition number of the covariance matrix of the Gaussian kernel:

Corollary 3.7 Consider the Gaussian kernel in (3.1) and $X _ { n + 1 } = X _ { n } \cup \left\{ x _ { n + 1 } \right\}$ . Then

$$
\mathrm { c o n d } ( \mathsf { K } _ { n + 1 } ) \geq \frac { 1 } { 2 } \mathopen { } \mathclose \bgroup \left( \frac { \sqrt { 2 } \left( b - a \right) } { \ell } \aftergroup \egroup \right) ^ { - 2 n } n ! \quad a n d \quad \mathrm { c o n d } ( \mathsf { K } _ { n + 1 } ) \geq \frac { 1 } { 2 d } \mathopen { } \mathclose \bgroup \left( \frac { \sqrt { 2 } \left( b - a \right) } { \ell } \aftergroup \egroup \right) ^ { - 2 m } m !
$$

in the settings of Theorem 3.1 $( d = 1 )$ and Theorem ${ \mathcal { 3 . 2 } } \ ( d \geq 1 )$ , respectively.

The condition number thus grows with a factorial rate. This means that the limited numerical precision prevents the implementation of Gaussian process interpolation if the Gaussian kernel is used. In the dominant double-precision floating-point arithmetic a condition number of about $1 0 ^ { 1 6 }$ is the most that can be handled without a loss of significant precision. In Figure 3 we saw that surprisingly few points are needed to lose numerical precision. The condition number cond $. ( \mathsf { K } _ { n } ) \approx 1 0 ^ { 1 6 }$ was exceeded with only $n = 8$ points. The same figure showed that finitely smooth Mat´erns are much less brittle in this sense. Next we explain why Mat´erns are diferent.

## 4. Smoothness matters

Many commonly used kernels are stationary, which means that $k ( x , y ) = \Phi ( x - y )$ for a function $\Phi \colon \mathbb { R } ^ { d }  \mathbb { R }$ and all $x , y \in \mathbb { R } ^ { d }$ . A stationary kernel is strictly positive-definite if the Fourier transform of $\Phi .$ , or the spectral density,

$$
( \mathcal { F } \Phi ) ( \omega ) = \frac { 1 } { ( 2 \pi ) ^ { d / 2 } } \int _ { \mathbb { R } ^ { d } } \Phi ( x ) e ^ { - i x \cdot \omega } \mathrm { d } x
$$

exists and is everywhere positive. The Fourier transform is real because $\Phi$ is symmetric about the origin. The rate of decay of the spectral density determines the smoothness of the kernel. In the univariate case,

$$
\Phi ^ { ( 2 m ) } ( 0 ) = \frac { 1 } { \sqrt { 2 \pi } } ( - 1 ) ^ { m } \int _ { \mathbb { R } } \omega ^ { 2 m } ( \mathcal { F } \Phi ) ( \omega ) \mathrm { d } \omega ,
$$

which exists only if the mapping $\omega \mapsto \omega ^ { 2 m } ( \mathcal { F } \Phi ) ( \omega )$ is integrable. The larger m is, the faster the spectral density must tend to zero as $| \omega |  \infty$ if this mapping is to be integrable. That is, the more diferentiable one wishes the kernel to be, the faster its spectral density must decay. The spectral density

$$
( \mathcal { F } \Phi ) ( \omega ) = \ell ^ { d } \exp ( - \ell ^ { 2 } \| \omega \| _ { 2 } ^ { 2 } / 2 )
$$

of the Gaussian kernel (3.1) is Gaussian and thus decays faster than any polynomial, meaning that the Gaussian kernel is infinitely diferentiable (a fact for which it is of course wholly unnecessary to invoke Fourier analysis).

## 4.1 Mat´erns and finite smoothness

In contrast to the Gaussian, the spectral density of a Mat´ern kernel

$$
k _ { \nu } ( x , y ) = \Phi _ { \nu } ( x - y ) = \frac { 2 ^ { 1 - \nu } } { \Gamma ( \nu ) } \left( \frac { \sqrt { 2 \nu } \left\| x - y \right\| _ { 2 } } { \ell } \right) ^ { \nu } \mathcal { K } _ { \nu } \left( \frac { \sqrt { 2 \nu } \left\| x - y \right\| _ { 2 } } { \ell } \right)
$$

on $\mathbb { R } ^ { d }$ has polynomial decay (Wendland, 2005, Theorem 6.13):

$$
( \mathcal { F } \Phi _ { \nu } ) ( \omega ) = \frac { 2 ^ { d / 2 } \Gamma ( \nu + d / 2 ) } { \Gamma ( \nu ) } \bigg ( \frac { 2 \nu } { \ell ^ { 2 } } \bigg ) ^ { \nu } \bigg ( \frac { 2 \nu } { \ell ^ { 2 } } + \| \omega \| _ { 2 } ^ { 2 } \bigg ) ^ { - ( \nu + d / 2 ) } .\tag{4.1}
$$

There is a sharp contrast between the behaviour of the conditional variance for the Gaussian kernel and kernels whose spectral density decays polynomially.

Theorem 4.1 Consider a stationary strictly positive-definite kernel $k ( x , y ) = \Phi ( x - y )$ for a continuous and integrable $\Phi \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ . Suppose that there are $c > 0$ and $s > 0$ such that $( \mathcal { F } \Phi ) ( \omega ) \geq c ( 1 + \| \omega \| _ { 2 } ^ { 2 } ) ^ { - s }$ for all $\omega \in \mathbb { R } ^ { d }$ . Let $\sigma \geq 0$ and $X _ { n } = \{ x _ { 1 } , \ldots , x _ { n } \} \subset \mathbb { R } ^ { d }$ . If $\begin{array} { r } { \delta ( x , X _ { n } ) = \operatorname* { m i n } _ { x _ { i } \in X _ { n } } \| x - x _ { i } \| _ { 2 } \leq 1 } \end{array}$ , then

$$
\mathbb { V } _ { \sigma } ( x \mid \mathcal { D } _ { n } ) \geq C \delta ( x , X _ { n } ) ^ { 2 s - d } ,\tag{4.2}
$$

where $C > 0$ is a constant that does not depend on $\sigma , x ,$ , or $X _ { n }$ .

![](images/9ac0dad449f138b9d5bc7f82db01f7ee8e2cb314e66935fe5d891fa50c900d7e.jpg)  
Figure 10: Infinitely diferentiable but non-analytic bump functions on R and $\mathbb { R } ^ { 2 }$

Proof See Section 7.3.

The quantity $\delta ( x , X _ { n } )$ appearing in (4.2) is the distance from x to the nearest point in $X _ { n }$ . When $s > d / 2$ , a condition that any valid Mat´ern kernel satisfies, this theorem states that the variance at a point x can tend to zero as $n \to \infty$ only if there are input points arbitrarily close to x. This is in stark contrast to Theorems 3.1 and 3.2 in which the variance for the Gaussian kernel tends to zero (and does so extremely fast) even if the input points are far away from x. These theorems make no reference to $\delta ( x , X _ { n } )$ or any other notions of closeness of x to $X _ { n }$ . The fundamental reason for this drastic diference is that the Gaussian kernel is analytic while Mat´erns are not.

## 4.2 Analytic functions and kernels

An infinitely diferentiable function $f \colon D \to \mathbb { R }$ on an open set $D \subset \mathbb { R } ^ { d }$ is real analytic if for every $\boldsymbol { x } _ { 0 } \in D$ there is a neighbourhood $A \subset D$ of $x _ { 0 }$ such that the Taylor series

$$
T _ { x _ { 0 } } ( x ) = \sum _ { \alpha \in \mathbb { N } _ { 0 } ^ { d } } \frac { \mathrm { D } ^ { \alpha } f ( x _ { 0 } ) } { \alpha ! } ( x - x _ { 0 } ) ^ { \alpha }
$$

converges to $f ( x )$ for every $x \in A .$ . We use standard multi-index notations $\boldsymbol { x } ^ { \alpha } = x _ { 1 } ^ { \alpha _ { 1 } } \cdot \cdot \cdot x _ { d } ^ { \alpha _ { d } }$ $\alpha ! = \alpha _ { 1 } ! \cdots \alpha _ { d } !$ , and $\mathrm { D } ^ { \alpha } f ( x ) = \partial _ { 1 } ^ { \alpha _ { 1 } } \cdot \cdot \cdot \partial _ { d } ^ { \alpha _ { d } } f ( x )$ . Every real analytic function is infinitely diferentiable (for otherwise the Taylor series could not be defined) but an infinitely diferentiable function need not be real analytic. The standard example of such a function is the bump function given by

$$
f ( x ) = \left\{ \begin{array} { l l } { \exp \left( - \displaystyle \frac { 1 } { 1 - \| x \| _ { 2 } ^ { 2 } } \right) } & { \mathrm { ~ i f ~ } \quad \| x \| _ { 2 } < 1 , } \\ { 0 } & { \mathrm { ~ i f ~ } \quad \| x \| _ { 2 } \geq 1 . } \end{array} \right.
$$

Figure 10 shows other examples of infinitely diferentiable yet non-analytic functions. An infinitely diferentiable function is real analytic if and only if $\mathrm { D } ^ { \alpha } f ( x )$ does not grow too fast as α increases. Namely, if for every compact $K \subset D$ there is $C \geq 0$ such that

$$
\operatorname* { s u p } _ { x \in K } \lvert \mathrm { D } ^ { \alpha } f ( x ) \rvert \leq C ^ { | \alpha | + 1 } \alpha ! \quad \mathrm { ~ f o r ~ e v e r y ~ } \quad \alpha \in \mathbb { N } _ { 0 } ^ { d } ,\tag{4.3}
$$

then $f$ is real analytic and vice versa. A crucial property of non-trivial real analytic functions is that they cannot vanish on large sets. See Krantz and Parks (2002, Cor. 1.2.7) and Mityagin (2020) for the following theorem.<sup>5</sup>

Theorem 4.2 (Identity theorem) Let $f \colon D \to \mathbb { R }$ be a real-analytic function on an open and connected set $D \subset \mathbb { R } ^ { d }$ . If f vanishes on an open subset of D, then $f = 0$ on D.

An immediate corollary of the identity theorem is that two real analytic functions that coincide on an open and connected set must coincide everywhere. In other words, a real analytic function is fully determined by its values on any given open set. This explains the observations in Section 3: With the Gaussian kernel the latent function is modelled as a real analytic function and thus having observations in any given open subset sufices to fully learn it. All analytic kernels induce similar behaviour.

A square-integrable function $f$ is real analytic if its Fourier transform tends to zero exponentially fast, meaning that there is $a > 0$ such that $\begin{array} { r } { ( \mathcal { F } f ) ( \omega ) = O ( \exp ( - a \| \omega \| _ { 2 } ) ) } \end{array}$ . This is easy to verify with the help of the Fourier identity $( { \mathcal { F } } \mathrm { D } ^ { \alpha } f ) ( \omega ) = ( i \omega ) ^ { \alpha } ( { \mathcal { F } } f ) ( \omega )$ and (4.3). A more general version of this result is known as the Paley–Wiener theorem (Reed and Simon, 1975, Thm. IX.13). For stationary kernels we obtain the following theorem, which states that the conditional variance tends to zero everywhere essentially regardless of the distribution of the input points if the spectral density decays exponentially.

Theorem 4.3 Consider a stationary strictly positive-definite kernel $k ( x , y ) = \Phi ( x - y )$ for a continuous and integrable $\Phi \colon { \mathbb { R } ^ { d } }  { \mathbb { R } }$ . Suppose that there are $C \geq 0$ and $a > 0$ such that $\begin{array} { r } { ( \mathcal { F } \Phi ) ( \omega ) \leq C \exp ( - a \| \omega \| _ { 2 } ) } \end{array}$ for all $\omega \in \mathbb { R } ^ { d }$ . Let $\sigma \geq 0$ . If $X _ { n } = \{ x _ { 1 } , \ldots , x _ { n } \}$ for a sequence $\{ x _ { i } \} _ { i = 1 } ^ { \infty } \subset \mathbb { R } ^ { d }$ of pairwise distinct points whose closure has non-empty interior, then

$$
\operatorname* { s u p } _ { x \in K } \mathbb { V } _ { \sigma } ( x \mid \mathcal { D } _ { n } ) \to 0 \quad \mathrm { ~ } a s \quad n \to \infty \quad f o r \ a n y \ c o m p a c t \quad K \subset \mathbb { R } ^ { d } .
$$

Proof See Section 7.3.

See Sun and Zhou (2008) for related results. Theorem 4.3 is a generalisation of Theorem 3.5, a corresponding result for the Gaussian kernel. The message of Theorems 4.1 and 4.3 is that the rate of decay of the spectral density determines whether or not a Gaussian process model is liable to overconfidence resulting from defectively distributed input points. If the spectral density does not decay faster than some polynomial (e.g., Mat´erns), the conditional variance cannot tend to zero everywhere unless the input points are space-filling; if the spectral density decays at least exponentially, the variance tends to zero whenever the inputs cover some open set.

There are well-known kernels that lie on the “boundary” between the two cases. An inverse multiquadric kernel

$$
k ( x , y ) = \left( 1 + \frac { \| x - y \| _ { 2 } ^ { 2 } } { \ell ^ { 2 } } \right) ^ { - \beta / 2 }\tag{4.4}
$$

is strictly positive-definite and integrable on $\mathbb { R } ^ { d }$ whenever $\beta > d$ (Wendland, 2005, Thm. 6.13). The special case $\beta = 2$ gives rise to the inverse quadratic kernel

$$
k ( x , y ) = \bigg ( 1 + \frac { \| x - y \| _ { 2 } ^ { 2 } } { \ell ^ { 2 } } \bigg ) ^ { - 1 } .\tag{4.5}
$$

From (4.1) we see that the spectral density of a Mat´ern is an inverse multiquadric, so by Fourier inversion the spectral density of an inverse multiquadric is a Mat´ern. It is then straightforward to work out that the spectral density of an inverse multiquadric is

$$
( \mathcal { F } \Phi ) ( \omega ) = \ell ^ { d } \frac { 2 ^ { 1 - \beta / 2 } } { \Gamma ( \beta / 2 ) } ( \ell \| \omega \| _ { 2 } ) ^ { ( \beta - d ) / 2 } \mathcal { K } _ { ( \beta - d ) / 2 } ( \ell \| \omega \| _ { 2 } ) .
$$

Because the Bessel function has the asymptotics $\mathcal { K } _ { \nu } ( r ) = \Theta ( r ^ { - 1 / 2 } e ^ { - r } )$ as $r  \infty$ for any $\nu > 0$ (Wendland, 2005, Sec. 5.1), the spectral density satisfies

$$
\begin{array} { r } { ( \mathcal { F } \Phi ) ( \omega ) = \Theta ( \| \omega \| _ { 2 } ^ { ( \beta - d ) / 2 - 1 / 2 } \exp ( - \ell \| \omega \| _ { 2 } ) ) . } \end{array}
$$

The spectral density decays exponentially fast and the kernel falls under Theorem 4.3.

## 4.3 Infinitely smooth non-analytic kernels

Theorem 4.1 covers finitely smooth kernels and Theorem 4.3 real analytic kernels. Gevrey kernels are infinitely diferentiable but non-analytic and therefore fall between the two classes. To the best of our knowledge these kernels have not appeared before in the Gaussian process literature. We say that a stationary kernel on $\mathbb { R } ^ { d }$ is a Gevrey kernel of order $\eta \in ( 0 , 1 )$ if its spectral density is

$$
\begin{array} { r } { ( \mathcal { F } \Phi _ { \eta } ) ( \omega ) = C _ { \eta } \exp ( - \rho \| \omega \| _ { 2 } ^ { \eta } ) , } \end{array}
$$

where the constant $\begin{array} { r } { C _ { \eta } = ( 2 \pi ) ^ { d / 2 } / \int _ { \mathbb { R } ^ { d } } \exp ( - \rho \| \omega \| _ { 2 } ^ { \eta } ) } \end{array}$ dω is there to ensure that $\Phi _ { \eta } ( 0 ) = 1$ Note that, formally, $\eta = 1$ corresponds to an inverse multiquadric with $\beta = d + 1$ and $\eta = 2$ to the Gaussian kernel. Because spectral densities of Gevrey kernels decay faster than any polynomial, these kernels are infinitely diferentiable. However, Gevrey kernels are not analytic. One can verify this for example by studying how fast

$$
\Phi _ { \eta } ^ { ( 2 \alpha ) } ( 0 ) = \frac { C _ { \eta } } { ( 2 \pi ) ^ { d / 2 } } ( - 1 ) ^ { | \alpha | } \int _ { \mathbb { R } ^ { d } } \omega ^ { 2 \alpha } \exp ( - \rho \| \omega \| _ { 2 } ^ { \eta } ) \mathrm { d } \omega
$$

blows up as α increases and using the analyticity characterisation in (4.3). Gevrey kernels are closely related to Gevrey spaces of infinitely diferentiable but non-analytic functions (Rodino, 1993) that appear in the analysis of partial diferential equations. The following theorem shows that Gevrey kernels resemble Mat´erns in that the input points must be space-filling if the conditional variance is to vanish.

Theorem 4.4 Consider a Gevrey kernel with $\eta \in ( 0 , 1 )$ . Let $\sigma \geq 0$ and $X _ { n } = \{ x _ { 1 } , \ldots , x _ { n } \} \subset$ $\mathbb { R } ^ { d }$ $\begin{array} { r } { I f \delta ( x , X _ { n } ) = \operatorname* { m i n } _ { x _ { i } \in X _ { n } } \| x - x _ { i } \| _ { 2 } } \end{array}$ is suficiently small, then

$$
\mathbb { V } _ { \sigma } ( x \mid \mathcal { D } _ { n } ) \geq C \exp ( - c \delta ( x , X _ { n } ) ^ { - \kappa } ) ,
$$

where $C , c , \kappa > 0$ are constants that do not depend on σ, x, or $X _ { n }$ .

![](images/06f4154744343495cf83f7fce8a70c8580a9d0161f2b05cc5f79014172f6e875.jpg)  
Figure 11: The conditional mean (black) and the pointwise 95% credible intervals (shaded) obtained from the conditional variance over the interval [0, 1] for the $\mathrm { { M a t } \acute { e } r n \ – 3 / 2 } .$ Whittaker, inverse quadratic, and Gaussian kernels with $\ell = 0 . 3$ . The data consist of four observations located on the first half of the interval that are indicated in red. Despite being infinitely diferentiable, the Whittaker and inverse quadratic kernels yield much larger variances than the Gaussian on the second half of the interval that contains no input points.

## Proof See Section 7.3.

Unfortunately, not many Gevrey kernels are available in closed form. A few somewhat tractable special cases exist because it turns out that Gevrey kernels correspond to density functions of stable distributions (Zolotarev, 1986; Garoni and Frankel, 2002). Let $d = 1$ The simplest special case is $\eta = 2 / 3$ . The resulting Whittaker kernel is defined by

$$
\Phi _ { \eta = 2 / 3 } ( r ) = \frac { 2 } { 3 \sqrt { 3 } \gamma | r | } \exp { \left( \frac { 2 } { 2 7 \gamma ^ { 2 } r ^ { 2 } } \right) } \mathcal { W } _ { - 1 / 2 , 1 / 6 } \left( \frac { 4 } { 2 7 \gamma ^ { 2 } r ^ { 2 } } \right) .
$$

Here $\gamma = \rho ^ { - 1 / \eta }$ and $\mathcal { W } _ { \kappa , \mu }$ is the Whittaker function, which has the expression

$$
\mathcal { W } _ { \kappa , \mu } ( z ) = \frac { z ^ { \kappa } e ^ { - z / 2 } } { \Gamma ( 1 / 2 - \kappa + \mu ) } \int _ { 0 } ^ { \infty } t ^ { - \kappa - 1 / 2 + \mu } \biggl ( 1 + \frac { t } { z } \biggr ) ^ { \kappa - 1 / 2 + \mu } e ^ { - t } \mathrm { d } t
$$

when $\kappa - 1 / 2 - \mu \in \mathbb { C }$ is not an integer and has non-positive real part. Figure 11 compares the conditional variance for the Whittaker kernel to a few popular kernels.

## 5. Apology of the Gaussian kernel

Finally, it seems fitting to review some of the reasons, post hoc or not, that have made the Gaussian kernel popular, or at the very least may have played a part in keeping it popular.

## 5.1 The Gaussian kernel is Gaussian

No doubt the main cause for the popularity of the Gaussian kernel is the fact that it is nothing but the Gaussian function that every student is familiar with. Its algebraic form is simple and convenient. No Bessel functions or other objects more advanced than those encountered on basic calculus courses are required for implementation and superficial understanding. Moreover, because the Gaussian function is infinitely diferentiable (and diferentiating it is relatively easy) and decays very fast, one never has to worry about diferentiability, integrability, or other such pesky technicalities. As its derivatives are straightforward to compute, the Gaussian kernel can be easily applied to modelling with derivative data. That many integrals involving the Gaussian kernel can be computed explicitly makes it a convenient kernel to use when working with Hilbert space embeddings of distributions (Muandet et al., 2017).

## 5.2 The only factorisable isotropic kernel

A kernel on $\mathbb { R } ^ { d }$ is isotropic if there is a function $\phi \colon [ 0 , \infty )  \mathbb { R }$ such that $k ( x , y ) = \phi ( \| x - y \| _ { 2 } )$ for all $x , y \in \mathbb { R } ^ { d }$ . That is, an isotropic kernel only depends on the distance between the inputs. Often one has to decide whether to use an isotropic kernel or a product-isotropic kernel

$$
k ( x , y ) = k _ { 1 } ( x _ { 1 } , y _ { 1 } ) \times \cdots \times k _ { d } ( x _ { d } , y _ { d } ) = \phi _ { 1 } ( | x _ { 1 } - y _ { 1 } | ) \times \cdots \times \phi _ { d } ( | x _ { d } - y _ { d } | ) ,
$$

where $k _ { i } ( x _ { i } , y _ { i } ) = \phi _ { i } ( | x _ { i } - y _ { i } | )$ are isotropic kernels on R. When using the Gaussian kernel one does not have to decide, for it factorises as a product of its one-dimensional versions:

$$
k ( x , y ) = \exp { \left( - { \frac { \| x - y \| _ { 2 } ^ { 2 } } { 2 \ell ^ { 2 } } } \right) } = \exp { \left( - { \frac { ( x _ { 1 } - y _ { 1 } ) ^ { 2 } } { 2 \ell ^ { 2 } } } \right) } \times \dots \times \exp { \left( - { \frac { ( x _ { d } - y _ { d } ) ^ { 2 } } { 2 \ell ^ { 2 } } } \right) } .
$$

It would be trivial to include and learn a diferent lengthscale for each dimension, which is sometimes referred to as automatic relevance determination. This property is unique to the Gaussian kernel as under very mild assumptions there are no other isotropic kernels that are coordinate-wise products of their one-dimensional versions (Stein, 1999, pp. 55–56).<sup>6</sup>

## 5.3 Simple series expansion

Access to a series expansion of the form $\begin{array} { r } { k ( x , y ) = \sum _ { p = 0 } ^ { \infty } \psi _ { p } ( x ) \psi _ { p } ( y ) } \end{array}$ is useful, be it to construct low-rank approximations by truncating the expansion (Solin and S¨arkk¨a, 2020) or for the purposes of theory (Minh, 2010). The Gaussian kernel has the simplest series expansion among commonly used stationary kernels, trivially derived from the Taylor expansion of the exponential function (here we set $d = 1$ for simplicity):

$$
\begin{array} { c } { \displaystyle { k ( x , y ) = \exp \left( - \frac { ( x - y ) ^ { 2 } } { 2 \ell ^ { 2 } } \right) = e ^ { - x ^ { 2 } / ( 2 \ell ^ { 2 } ) } e ^ { x y / \ell ^ { 2 } } e ^ { - y ^ { 2 } / ( 2 \ell ^ { 2 } ) } } } \\ { \displaystyle { = \sum _ { p = 0 } ^ { \infty } \left( \frac { x ^ { p } } { \ell ^ { p } \sqrt { p ! } } e ^ { - x ^ { 2 } / ( 2 \ell ^ { 2 } ) } \right) \left( \frac { y ^ { p } } { \ell ^ { p } \sqrt { p ! } } e ^ { - y ^ { 2 } / ( 2 \ell ^ { 2 } ) } \right) . } } \end{array}\tag{5.1}
$$

The Gaussian also has a known Mercer expansion with respect to the normal distribution. Though the Mercer expansion is more complicated than the expansion in (5.1), it only

requires Hermite polynomials and exponentials. Series expansions, Mercer or not, for other stationary kernels tend to be much more involved; examples can be found in Tronarp and Karvonen (2024).

## 5.4 Limiting case in the Mat´ern class

The parameter $\nu > 0$ of the Mat´ern class

$$
k _ { \nu } ( x , y ) = \frac { 2 ^ { 1 - \nu } } { \Gamma ( \nu ) } \left( \frac { \sqrt { 2 \nu } \left\| x - y \right\| _ { 2 } } { \ell } \right) ^ { \nu } \mathcal { K } _ { \nu } \left( \frac { \sqrt { 2 \nu } \left\| x - y \right\| _ { 2 } } { \ell } \right)
$$

controls the diferentiability of the kernel and the corresponding Gaussian process. It is well known that $k _ { \nu }$ tends pointwise to the Gaussian kernel as $\nu \to \infty$ (Stein, 1999, pp. 49–50).<sup>7</sup> It tends to be dificult to estimate diferentiability from finite data. In using the Gaussian kernel one in essence removes $\nu ,$ a dificult-to-specify parameter, from the Mat´ern model. As discussed in Section 5.2, this also removes the need to decide between an isotropic and product-isotropic kernel. Because Mat´erns encode a full scale of Sobolev regularity assumptions, the Gaussian can be thought of as an “infinitely smooth Sobolev prior” (this statement is not fully devoid of mathematical meaning; see Novak et al., 2018).

## 5.5 Member of the γ-exponential class

Let $\gamma \in ( 0 , 2 ]$ . Then the γ-exponential kernel

$$
k _ { \gamma } ( x , y ) = \exp { \left( - \frac { \| x - y \| _ { 2 } ^ { \gamma } } { \gamma \ell ^ { \gamma } } \right) }\tag{5.2}
$$

is positive-definite. These kernels were popularised by Sacks et al. (1989) and often appear in emulation literature.<sup>8</sup> With parameters $\gamma = 2$ and $\gamma = 1$ we get the Gaussian and the Mat´ern-1/2 kernels $k _ { 2 } ( x , y ) = \exp ( - \| x - y \| _ { 2 } ^ { 2 } / ( 2 \ell ^ { 2 } ) )$ and $k _ { 1 } ( x , y ) = \exp ( { - \| x - y \| _ { 2 } / \ell } )$ The γ-exponential therefore provides another “natural” connection between Gaussians and finitely smooth kernels. However, when one looks any deeper it quickly becomes apparent that the connection is far from natural: For every $\gamma \in ( 0 , 2 )$ the γ-exponential is at most once diferentiable with respect to both arguments, yet becomes infinitely diferentiable when $\gamma = 2$ . That is, the class of kernels defined by (5.2) is continuous in $\gamma$ when it comes to the functional form of the kernel, but not when it comes to the properties of the kernel. The only advantage the γ-exponential class enjoys over Mat´erns in (2.2) is its simplicity (no Bessel functions).

## 5.6 Universality

One factor that may have led people to use the Gaussian kernel is that it is universal. A kernel on a compact $D \subset \mathbb { R } ^ { d }$ is universal if its reproducing kernel Hilbert space (see

Section 7.1) is dense in $C ( D )$ , the space of continuous functions on D. A universal kernel can therefore be used to approximate any continuous function, and this is a desirable property when no strong structural information is available on the unknown function being modelled. However, universality has little use as a kernel decision criterion among kernels in everyday usage because every commonly used kernel is universal; for example, all stationary kernels that appear in this article are universal (Micchelli et al., 2006, Thm. 17).

## 5.7 When to use the Gaussian

The central message of this article is that the Gaussian kernel is not the kernel to default to. It should be used only if there are compelling reasons to believe that the latent phenomenon one models is extremely smooth. And even then it may be safer to use other infinitely smooth kernels, such as inverse multiquadrics or Gevreys. The Gaussian kernel is far less dangerous in regression $( \sigma > 0 )$ than in interpolation $( \sigma = 0 )$ , where it can induce catastrophic failure of both numerics and uncertainty quantification. Nevertheless, even in regression one should never intentionally use a model that is known to be further from the truth than almost any other standard model. It should be noted that we have focussed on Gaussian processes, a perspective we admit is quite narrow. There are related tasks in which the Gaussian kernel is known to shine. For example, the support vector machine based on the Gaussian kernel is rate-adaptive over Sobolev spaces: It achieves (up to a sub-polynomial term) the minimax optimal learning rate $n ^ { - 2 s / ( 2 \bar { s } + d ) }$ in a Sobolev space of order s on a subset of $\mathbb { R } ^ { d }$ if the lengthscale is selected in an appropriate data-driven way (Eberts and Steinwart, 2013).

## 6. What about kernel learning?

We have but touched upon kernel learning, the data-driven choice of the kernel. Tuning a few parameters of a standard parametric family of kernels (e.g., the lengthscale as discussed in Remark 3.3) is often too inflexible. Composing functions and kernels (Lloyd et al., 2014; Wilson et al., 2016) and learning a kernel in the spectral domain (L´azaro-Gredilla et al., ${ 2 0 1 0 } ;$ Remes et al., 2017) are the two main approaches to flexible kernel learning. Our results as such do not apply to kernel learning. However, our work opens an interesting research direction to inform the design of kernel learning architectures. Let us consider deep kernel learning (Wilson et al., 2016) as a concrete example. In deep kernel learning, flexibility is achieved by using a composite kernel

$$
k ( x , y ) = k _ { 0 } ( g ( x , w ) , g ( y , w ) ) ,
$$

where $k _ { 0 }$ is a potentially parametric base kernel and g a neural network with weights or other parameters w. The base kernel $k _ { 0 }$ is often Gaussian, and data are used to learn the weights and other parameters. As compositions of analytic functions are analytic, our work, when combined with the well-understood properties of composite kernels (Paulsen and Raghupathi, 2016, Ch. 5), suggests that it may be best to avoid analytic activation functions (e.g., the swish function) in the deep kernel learning framework of Wilson et al. (2016).

## 7. Proofs

This section contains the proofs of the theorems appearing in this article.

## 7.1 Reproducing kernel Hilbert spaces

Every positive-semidefinite kernel k : $D \times D \to \mathbb { R }$ on a set D induces a unique reproducing kernel Hilbert space (RKHS). This Hilbert space, denoted $H ( k )$ , consists of certain functions $f \colon D  \mathbb { R }$ and is characterised by $k ( \cdot , x ) \in H ( k )$ for every $x \in D$ and the important reproducing property:

$$
f ( x ) = \langle f , k ( \cdot , x ) \rangle _ { H ( k ) } \quad { \mathrm { ~ f o r ~ a l l ~ } } \quad f \in H ( k ) { \mathrm { ~ a n d ~ } } x \in D .\tag{7.1}
$$

We refer to Berlinet and Thomas-Agnan (2004) and Paulsen and Raghupathi (2016) for the theory of RKHSs. From the functional form of the kernel alone it tends to be dificult to deduce what $H ( k )$ and its inner product look like. Fortunately, if the kernel is stationary on $\mathbb { R } ^ { d }$ , meaning that $k ( x , y ) = \Phi ( x - y )$ for $\Phi \colon \mathbb { R } ^ { d }  \mathbb { R }$ , and $\Phi$ is continuous and integrable, the RKHS admits the convenient spectral characterisation

$$
H ( k ) = \left\{ f \in L ^ { 2 } ( { \mathbb { R } } ^ { d } ) : \| f \| _ { H ( k ) } ^ { 2 } = { \frac { 1 } { ( 2 \pi ) ^ { d / 2 } } } \int _ { { \mathbb { R } } ^ { d } } { \frac { | ( { \mathcal { F } } f ) ( \omega ) | ^ { 2 } } { ( { \mathcal { F } } \Phi ) ( \omega ) } } \mathrm { d } \omega < \infty \right\}\tag{7.2}
$$

in terms of the spectral density, FΦ (recall Section 4). See Theorem 10.12 in Wendland (2005) or Kimeldorf and Wahba (1970) for the spectral characterisation. The faster the spectral density tends to zero $( \mathbf { i . e . }$ , the smoother the kernel), the smaller the RKHS.

The proofs are based on the following equivalence between the conditional variance and worst-case error. More such equivalences can be found in Chapters 5 and 6 of Kanagawa et al. (2025). We write $k ^ { x } = k ( \cdot , x )$ for a kernel translate.

Lemma 7.1 Let $\sigma \geq 0$ . Then

$$
\mathbb { V } _ { \sigma } ( x \mid \mathcal { D } _ { n } ) = \operatorname* { m i n } _ { \mathbf { u } \in \mathbb { R } ^ { n } } \left\{ \left\| k ^ { x } - \sum _ { i = 1 } ^ { n } u _ { i } k ^ { x _ { i } } \right\| _ { H ( k ) } ^ { 2 } + \sigma ^ { 2 } \left\| \mathbf { u } \right\| _ { 2 } ^ { 2 } \right\}\tag{7.3}
$$

for every $x \in D$ . Moreover, $\mathbb { V } _ { \sigma } ( x \mid \mathcal { D } _ { n + 1 } ) \le \mathbb { V } _ { \sigma } ( x \mid \mathcal { D } _ { n } )$ for every $n \geq 1$ and $x \in D$

Proof By the reproducing property (7.1), $\langle k ^ { x } , k ^ { y } \rangle _ { H ( k ) } = k ( x , y )$ , so that expanding the square in (7.3) gives the quadratic objective

$$
\begin{array} { r } { J _ { n } ( \mathbf { u } ) = k ( x , x ) - 2 \mathbf { u } ^ { \intercal } \mathbf { k } _ { n } ( x ) + \mathbf { u } ^ { \intercal } \mathsf { K } _ { n } \mathbf { u } + \sigma ^ { 2 } \Vert \mathbf { u } \Vert _ { 2 } ^ { 2 } = k ( x , x ) - 2 \mathbf { u } ^ { \intercal } \mathbf { k } _ { n } ( x ) + \mathbf { u } ^ { \intercal } ( \mathsf { K } _ { n } + \sigma ^ { 2 } \mathsf { l } _ { n } ) \mathbf { u } . } \end{array}
$$

Because ${ \sf K } _ { n } + \sigma ^ { 2 } { \sf I } _ { n }$ is a positive-definite matrix, $J _ { n }$ is strictly convex and its unique minimiser is $\mathsf { u } _ { * } = ( \mathsf { K } _ { n } + \sigma ^ { 2 } \mathsf { I } _ { n } ) ^ { - 1 } \mathsf { k } _ { n } ( x )$ , for which

$$
J _ { n } ( \mathbf { u } _ { * } ) = k ( x , x ) - \mathbf { k } _ { n } ( x ) ^ { \top } ( \mathsf { K } _ { n } + \sigma ^ { 2 } \mathsf { l } _ { n } ) ^ { - 1 } \mathbf { k } _ { n } ( x ) = \mathbb { V } _ { \sigma } ( x \mid \mathcal { D } _ { n } )
$$

by (2.3). Monotonicity in n follows from $J _ { n } ( \mathbf { u } ) = J _ { n + 1 } ( ( \mathbf { u } , 0 ) )$

The reproducing property can be utilised to rewrite the variance as a worst-case error. Lemma 7.2 Let $\sigma \geq 0$ . Then

$$
\mathbb { V } _ { \sigma } ( x \mid \mathcal { D } _ { n } ) = \operatorname* { m i n } _ { \mathbf { u } \in \mathbb { R } ^ { n } } \left\{ \operatorname* { s u p } _ { \| f \| _ { H ( k ) } \leq 1 } \left| f ( x ) - \sum _ { i = 1 } ^ { n } f ( x _ { i } ) u _ { i } \right| ^ { 2 } + \sigma ^ { 2 } \| \mathbf { u } \| _ { 2 } ^ { 2 } \right\} .\tag{7.4}
$$

Proof That the right-hand side in (7.4) does not exceed that in (7.3) follows directly from the reproducing property and the Cauchy–Schwarz inequality. That we obtain an equality is verified by considering the function $\begin{array} { r } { f = \big ( k ^ { x } - \textstyle \sum _ { i = 1 } ^ { n } u _ { i } k ^ { x _ { i } } \big ) / \| k ^ { x } - \textstyle \sum _ { i = 1 } ^ { n } u _ { i } k ^ { x _ { i } } \| _ { H ( k ) } } \end{array}$ , which has unit norm in $H ( k )$ . For details, see for example Sections 5.4 and 6.6 in Kanagawa et al. (2025).

The following results are also useful.

Lemma 7.3 If $\sigma \ge \varsigma \ge 0$ , then $\mathbb { V } _ { \sigma } ( x \mid \mathcal { D } _ { n } ) \ge \mathbb { V } _ { \varsigma } ( x \mid \mathcal { D } _ { n } )$ for every $x \in D$

Proof Because $\mathsf { K } _ { n } + \varsigma ^ { 2 } \mathsf { I } _ { n } \leq \mathsf { K } _ { n } + \sigma ^ { 2 } \mathsf { I } _ { n } \mathrm { i f } \sigma \geq \varsigma$ in the Loewner ordering of positive-semidefinite matrices, the claim follows from the expression for the variance in (2.3). ■

In the next proposition, which appears as Corollary 4.36 in Steinwart and Christmann (2008), we use the notation

$$
k ^ { \alpha , \beta } ( x , y ) = \mathrm { D } _ { v } ^ { \alpha } \mathrm { D } _ { w } ^ { \beta } k ( v , w ) \Big | _ { { v = x } \atop w = y }
$$

for kernel derivatives. Here the subscripts denote the variable with respect to which the kernel is being diferentiated.

Proposition 7.4 If k is a positive-semidefinite kernel on an open set $D \subset \mathbb { R } ^ { d }$ such that $k ^ { \alpha , \alpha }$ exists and is continuous for every $\alpha \in  { \mathbb { N } } _ { 0 } ^ { d }$ such that $| \alpha | \leq m _ { \mathrm { : } }$ , then every $f \in H ( k )$ is m times continuously diferentiable and

$$
\mathrm { D } ^ { \alpha } f ( x ) = \langle f , \mathrm { D } _ { x } ^ { \alpha } k ( \cdot , x ) \rangle _ { H ( k ) }\tag{7.5}
$$

for any $\alpha \in  { \mathbb { N } } _ { 0 } ^ { d }$ such that $| \alpha | \leq m$ and $x \in D$

## 7.2 Proofs for Section 3

This section contains the proofs for Section 3.

Proof of Theorem 3.1 Let

$$
( P _ { n } f ) ( x ) = \sum _ { i = 1 } ^ { n } f ( x _ { i } ) \prod _ { j \neq i } { \frac { x - x _ { j } } { x _ { i } - x _ { j } } }\tag{7.6}
$$

be the unique polynomial of degree at most $n - 1$ that interpolates f at our pairwise distinct points $x _ { 1 } , \ldots , x _ { n } \in D = [ a , b ]$ . It is a basic result of polynomial interpolation that for each $x \in D$ there is $\xi _ { x } \in D$ such that

$$
f ( x ) - ( P _ { n } f ) ( x ) = { \frac { f ^ { ( n ) } ( \xi _ { x } ) } { n ! } } \prod _ { i = 1 } ^ { n } ( x - x _ { i } )
$$

if f is n times continuously diferentiable. Consequently,

$$
| f ( x ) - ( P _ { n } f ) ( x ) | \leq { \frac { \operatorname* { s u p } _ { x \in D } | f ^ { ( n ) } ( x ) | } { n ! } } \operatorname* { s u p } _ { x \in D } \left| \prod _ { i = 1 } ^ { n } ( x - x _ { i } ) \right| \leq { \frac { \operatorname* { s u p } _ { x \in D } | f ^ { ( n ) } ( x ) | } { n ! } } ( b - a ) ^ { n }
$$

for every $x \in D$ . Every element of the RKHS of the Gaussian kernel is infinitely diferentiable by Proposition 7.4. Recall that we consider the case $\sigma = 0$ . From (7.3) and the form of the polynomial interpolant in (7.6) it follows that

$$
\begin{array} { l } { \displaystyle \operatorname* { s u p } _ { x \in D } \mathbb { V } ( x \mid \mathcal { D } _ { n } ) \leq \displaystyle \operatorname* { s u p } _ { x \in D } \ \operatorname* { s u p } _ { \| f \| _ { H ( k ) } \leq 1 } | f ( x ) - ( P _ { n } f ) ( x ) | ^ { 2 } } \\ { \leq \left( \frac { ( b - a ) ^ { n } } { n ! } \right) ^ { 2 } \displaystyle \operatorname* { s u p } _ { \| f \| _ { H ( k ) } \leq 1 } \displaystyle \operatorname* { s u p } _ { x \in D } \left| f ^ { ( n ) } ( x ) \right| ^ { 2 } . } \end{array}\tag{7.7}
$$

It is therefore necessary to bound the nth derivative of a function in $H ( k )$ . Applying the Cauchy–Schwarz inequality to (7.5) gives

$$
\left| f ^ { ( n ) } ( x ) \right| = \left| \left. f , \frac { \partial ^ { n } } { \partial x ^ { n } } k ( \cdot , x ) \right. _ { H ( k ) } \right| \leq \| f \| _ { H ( k ) } \biggl \| \frac { \partial ^ { n } } { \partial x ^ { n } } k ( \cdot , x ) \biggr \| _ { H ( k ) }\tag{7.8}
$$

for every $x \in D$ . Now,

$$
\left. \frac { \partial ^ { n } } { \partial x ^ { n } } k ( \cdot , x ) \right. _ { H ( k ) } ^ { 2 } = \frac { \partial ^ { 2 n } } { \partial v ^ { n } \partial w ^ { n } } k ( v , w ) \Bigg | _ { { v = x } } = ( - 1 ) ^ { n } \frac { \mathrm { d } ^ { 2 n } } { \mathrm { d } z ^ { 2 n } } \exp \left( - \frac { z ^ { 2 } } { 2 \ell ^ { 2 } } \right) \Bigg | _ { z = 0 } ,
$$

where the first equality is valid for any suficiently smooth kernel and the second uses the stationarity of the Gaussian kernel. Straightforward diferentiation of the Taylor series for $\exp ( - z ^ { 2 } / ( 2 \ell ^ { 2 } ) )$ then gives

$$
( - 1 ) ^ { n } { \frac { \mathrm { d } ^ { 2 n } } { \mathrm { d } z ^ { 2 n } } } \exp \left( - \left. { \frac { z ^ { 2 } } { 2 \ell ^ { 2 } } } \right) \right| _ { z = 0 } = \ell ^ { - 2 n } { \frac { ( 2 n ) ! } { 2 ^ { n } n ! } } .
$$

Inserting this in (7.8) gives the bound

$$
\operatorname* { s u p } _ { x \in D } | f ^ { ( n ) } ( x ) | \leq \| f \| _ { H ( k ) } \ell ^ { - n } { \sqrt { \frac { ( 2 n ) ! } { 2 ^ { n } n ! } } }
$$

on derivatives of elements of the Gaussian RKHS. Inserting this bound in (7.7) and applying the non-asymptotic version of Stirling’s approximation (Robbins, 1955),

$$
\sqrt { 2 \pi } n ^ { n + 1 / 2 } e ^ { - n } \leq n ! \leq \sqrt { 2 \pi } n ^ { n + 1 / 2 } e ^ { - n + 1 } ,\tag{7.9}
$$

to (2n)! and $( n ! ) ^ { 2 }$ yields

$$
\operatorname* { s u p } _ { x \in D } \mathbb { V } ( x \mid \mathcal { D } _ { n } ) \leq \left( \frac { b - a } { \ell } \right) ^ { 2 n } \frac { ( 2 n ) ! } { 2 ^ { n } ( n ! ) ^ { 2 } } \cdot \frac { 1 } { n ! } \leq \frac { e } { \sqrt { \pi n } } \Bigg ( \frac { \sqrt { 2 } ( b - a ) } { \ell } \Bigg ) ^ { 2 n } \frac { 1 } { n ! } .
$$

The claimed bound follows from $e / { \sqrt { \pi n } } \leq 2$

Proof of Theorem 3.2 Because D is bounded, it is contained in some hypercube $D ^ { \prime }$ Since $\begin{array} { r } { \operatorname* { s u p } _ { x \in D } \mathbb { V } ( x \mid \mathcal { D } _ { n } ) \leq \operatorname* { s u p } _ { x \in D ^ { \prime } } \mathbb { V } ( x \mid \mathcal { D } _ { n } ) } \end{array}$ , to derive an upper bound we may assume that

$D = [ a , b ] ^ { d }$ for some $a < b$ . Since the conditional variance cannot increase when new points are added, we may assume that $n = m ^ { d }$ and $X _ { n } = X _ { 1 } ^ { \prime } \times \cdot \cdot \cdot \times X _ { d } ^ { \prime }$ for $X _ { j } ^ { \prime } = \{ x _ { { j , 1 } } , \dotsc , x _ { { j , m } } \} \subset$ R such that $X _ { n } \subset D$ . Write $x = ( x _ { 1 } , \ldots , x _ { d } ) \in \mathbb { R } ^ { d }$ and $y = ( y _ { 1 } , \dots , y _ { d } ) \in \mathbb { R } ^ { d }$ . Because both the multivariate Gaussian kernel

$$
k ( x , y ) = \exp \left( - \frac { \| x - y \| _ { 2 } ^ { 2 } } { 2 \ell ^ { 2 } } \right) = \prod _ { j = 1 } ^ { d } r ( x _ { j } , y _ { j } ) : = \prod _ { j = 1 } ^ { d } \exp \left( - \frac { ( x _ { j } - y _ { j } ) ^ { 2 } } { 2 \ell ^ { 2 } } \right)
$$

and the points $X _ { n }$ have tensor product structure, we have

$$
{ \mathsf { K } } _ { n } = \mathsf { R } _ { 1 , m } \otimes \cdots \otimes \mathsf { R } _ { d , m } \in \mathbb { R } ^ { n \times n } \quad \mathrm { ~ a n d ~ } \quad \mathsf { k } _ { n } ( x ) = \mathsf { r } _ { 1 , m } ( x _ { 1 } ) \otimes \cdots \otimes \mathsf { r } _ { d , m } ( x _ { d } ) \in \mathbb { R } ^ { n } ,
$$

where $\otimes$ is the Kronecker product, $\mathsf { R } _ { j , m } = ( r ( x _ { j , i } , x _ { j , l } ) ) _ { i , l = 1 } ^ { m }$ , and $\mathfrak { r } _ { j , m } ( x _ { j } ) = ( r ( x _ { j } , x _ { j , i } ) ) _ { i = 1 } ^ { m }$ It now follows from properties of the Kronecker product that

$$
\begin{array} { l } { \displaystyle \mathbb { V } ( x \mid \mathcal { D } _ { n } ) = k ( x , x ) - \mathsf { k } _ { n } ( x ) ^ { \top } \mathsf { K } _ { n } ^ { - 1 } \mathsf { k } _ { n } ( x ) = \displaystyle \prod _ { j = 1 } ^ { d } r ( x _ { j } , x _ { j } ) - \prod _ { j = 1 } ^ { d } \mathsf { r } _ { j , m } ( x _ { j } ) ^ { \top } \mathsf { R } _ { j , m } ^ { - 1 } \mathsf { r } _ { j , m } ( x _ { j } ) } \\ { \displaystyle = : \prod _ { j = 1 } ^ { d } r _ { j } - \prod _ { j = 1 } ^ { d } q _ { j } . } \end{array}
$$

Denote $r _ { i : j } = r _ { i } \times \cdots \times r _ { j }$ and $q _ { i : j } = q _ { i } \times \cdots \times q _ { j }$ for $1 \leq i \leq j \leq d .$ Then iteration gives

$$
\mathbb { V } ( x \mid \mathcal { D } _ { n } ) = r _ { 1 } r _ { 2 : d } - q _ { 1 } q _ { 2 : d } = r _ { 2 : d } ( r _ { 1 } - q _ { 1 } ) + ( r _ { 2 : d } - q _ { 2 : d } ) q _ { 1 } = \sum _ { j = 1 } ^ { d } ( r _ { j } - q _ { j } ) r _ { j + 1 : d } \prod _ { i = 1 } ^ { j - 1 } q _ { i } ,
$$

where we use the convention $r _ { d + 1 : d } = 1$ . Because the conditional variance is non-negative, we have

$$
q _ { j } = \mathsf { r } _ { j , m } ( x _ { j } ) ^ { \top } \mathsf { R } _ { j , m } ^ { - 1 } \mathsf { r } _ { j , m } ( x _ { j } ) \leq r _ { j } = r ( x _ { j } , x _ { j } ) = 1 ,
$$

from which it follows that

$$
\mathbb { V } ( x \mid \mathcal { D } _ { n } ) \leq \sum _ { j = 1 } ^ { d } ( r _ { j } - q _ { j } ) = \sum _ { j = 1 } ^ { d } \left[ r ( x _ { j } , x _ { j } ) - \mathfrak { r } _ { j , m } ( x _ { j } ) ^ { \top } \mathsf { R } _ { j , m } ^ { - 1 } \mathsf { r } _ { j , m } ( x _ { j } ) \right] .\tag{7.10}
$$

Because $\boldsymbol { r }$ is the univariate Gaussian kernel and $X _ { i } ^ { \prime } \subset [ a , b ]$ for every $j ,$ , we may apply Theorem 3.1 with $n = m$ to each term in (7.10). This yields

$$
\mathbb { V } ( x \mid \mathcal { D } _ { n } ) \leq 2 \sum _ { j = 1 } ^ { d } \left( \frac { \sqrt { 2 } ( b - a ) } { \ell } \right) ^ { 2 m } \frac { 1 } { m ! } = 2 d \left( \frac { \sqrt { 2 } ( b - a ) } { \ell } \right) ^ { 2 m } \frac { 1 } { m ! } ,
$$

which completes the proof.

Proof of Proposition 3.4 By Lemma 7.3 it sufices to consider the case $\sigma = 0$ . The RKHS of the Gaussian kernel on $\mathbb { R } ^ { d }$ has the characterisation (Minh, 2010)

$$
H ( k ) = \Bigg \{ f ( x ) = e ^ { - \| x \| _ { 2 } ^ { 2 } / ( 2 \ell ^ { 2 } ) } \sum _ { \alpha \in \mathbb { N } _ { 0 } ^ { d } } c _ { \alpha } x ^ { \alpha } : \| f \| _ { H ( k ) } ^ { 2 } = \sum _ { \alpha \in \mathbb { N } _ { 0 } ^ { d } } \ell ^ { 2 | \alpha | } \alpha ! c _ { \alpha } ^ { 2 } < \infty \Bigg \} .\tag{7.11}
$$

Because $\{ x _ { i } \} _ { i = 1 } ^ { \infty } \subset R _ { 0 }$ , where $R _ { 0 }$ is the set of roots of some non-trivial d-variate polynomial $P$ and the function $g ( \acute { z } ) = e ^ { - \| z \| _ { 2 } ^ { 2 } / ( 2 \ell ^ { 2 } ) } P ( z )$ is in $H ( k )$ and has positive H(k)-norm by (7.11), we get from (7.4) and $g ( x _ { i } ) = 0$ for all $i \in \mathbb N$ that

$$
\mathbb { V } ( x \mid \mathcal { D } _ { n } ) = \operatorname* { m i n } _ { { \bf u } \in \mathbb { R } ^ { n } } \operatorname* { s u p } _ { \| f \| _ { { \cal H } ( k ) } \leq 1 } \left| f ( x ) - \sum _ { i = 1 } ^ { n } f ( x _ { i } ) u _ { i } \right| ^ { 2 } \geq \operatorname* { m i n } _ { { \bf u } \in \mathbb { R } ^ { n } } \frac { g ( x ) ^ { 2 } } { \| g \| _ { { \cal H } ( k ) } ^ { 2 } } = \frac { g ( x ) ^ { 2 } } { \| g \| _ { { \cal H } ( k ) } ^ { 2 } } > 0
$$

for every $n \in \mathbb { N } .$ . This is the claim.

## 7.3 Proofs for Section 4

This section contains the proofs for Section 4.

Proof of Theorem 4.1 By Lemma 7.3 it sufices to consider the case $\sigma = 0$ . Fix $\boldsymbol { x } \in \mathbb { R } ^ { d }$ and let $\begin{array} { r } { \delta = \delta ( x , X _ { n } ) = \operatorname* { m i n } _ { x _ { i } \in X _ { n } } \| x - x _ { i } \| _ { 2 } } \end{array}$ denote the distance of $_ x$ to $X _ { n }$ . Define the standard bump function $\varphi$ as

$$
\varphi ( y ) = \exp \left( - \frac { 1 } { 1 - \| y \| _ { 2 } ^ { 2 } } \right) \mathrm { i f } \| y \| _ { 2 } < 1 \mathrm { a n d } \varphi ( y ) = 0 \mathrm { o t h e r w i s e . }
$$

Because $\varphi$ is supported on the unit ball and $\| x - x _ { i } \| _ { 2 } \geq \delta$ for every $x _ { i } \in X _ { n }$ , the function $g = \varphi ( ( \cdot - x ) / \delta )$ vanishes on $X _ { n }$ and $g ( x ) = \varphi ( 0 ) = e ^ { - 1 }$ . The Fourier transform of $\varphi$ decays super-polynomially, so that $g \in H ( k )$ by the spectral characterisation (7.2). The worst-case interpretation (7.4) therefore gives

$$
\mathbb { V } ( x \mid \mathcal { D } _ { n } ) = \operatorname* { m i n } _ { \boldsymbol { \mathfrak { u } } \in \mathbb { R } ^ { n } } \operatorname* { s u p } _ { \Vert \boldsymbol { f } \Vert _ { H ( k ) } \leq 1 } \left. \boldsymbol { f } ( \boldsymbol { x } ) - \sum _ { i = 1 } ^ { n } \boldsymbol { f } ( \boldsymbol { x } _ { i } ) \boldsymbol { u } _ { i } \right. ^ { 2 } \geq \operatorname* { m i n } _ { \boldsymbol { \mathfrak { u } } \in \mathbb { R } ^ { n } } \frac { g ( \boldsymbol { x } ) ^ { 2 } } { \Vert \boldsymbol { g } \Vert _ { H ( k ) } ^ { 2 } } = \frac { e ^ { - 2 } } { \Vert \boldsymbol { g } \Vert _ { H ( k ) } ^ { 2 } } .
$$

We are thus left to estimate the RKHS norm of g from above. Assume that $\delta \leq 1$ . By the spectral characterisation (7.2) and the assumed lower bound on the spectral density,

$$
\begin{array} { r l } { \displaystyle \| g \| _ { H ( k ) } ^ { 2 } = \frac { 1 } { ( 2 \pi ) ^ { d / 2 } } \int _ { \mathbb { R } ^ { d } } \frac { | \mathcal { F } g ( \omega ) | ^ { 2 } } { \mathcal { F } \Phi ( \omega ) } \mathrm { d } \omega \leq \frac { 1 } { c ( 2 \pi ) ^ { d / 2 } } \int _ { \mathbb { R } ^ { d } } | \mathcal { F } g ( \omega ) | ^ { 2 } ( 1 + \| \omega \| _ { 2 } ^ { 2 } ) ^ { s } \mathrm { d } \omega } \\ & { \quad = \frac { \delta ^ { 2 d } } { c ( 2 \pi ) ^ { d / 2 } } \int _ { \mathbb { R } ^ { d } } | \mathcal { F } g ( \delta \omega ) | ^ { 2 } ( 1 + \| \omega \| _ { 2 } ^ { 2 } ) ^ { s } \mathrm { d } \omega } \\ & { \quad = \frac { \delta ^ { d } } { c ( 2 \pi ) ^ { d / 2 } } \int _ { \mathbb { R } ^ { d } } | \mathcal { F } \varphi ( \omega ) | ^ { 2 } \bigg ( 1 + \frac { \| \omega \| _ { 2 } ^ { 2 } } { \delta ^ { 2 } } \bigg ) ^ { s } \mathrm { d } \omega } \\ & { \quad = \frac { \delta ^ { d } } { c ( 2 \pi ) ^ { d / 2 } } \int _ { \mathbb { R } ^ { d } } | \mathcal { F } \varphi ( \omega ) | ^ { 2 } \delta ^ { - 2 s } ( \delta ^ { 2 } + \| \omega \| _ { 2 } ^ { 2 } ) ^ { s } \mathrm { d } \omega } \\ & { \quad \leq \frac { \delta ^ { d - 2 s } } { c ( 2 \pi ) ^ { d / 2 } } \int _ { \mathbb { R } ^ { d } } | \mathcal { F } \varphi ( \omega ) | ^ { 2 } ( 1 + \| \omega \| _ { 2 } ^ { 2 } ) ^ { s } \mathrm { d } \omega , } \end{array}
$$

where the integral is finite. Therefore $\| g \| _ { H ( k ) } ^ { 2 } \le C \delta ^ { d - 2 s }$ for a constant C that does not depend on x or $X _ { n }$ . It follows that

$$
\mathbb { V } ( x \mid \mathcal { D } _ { n } ) \geq \frac { e ^ { - 2 } } { \| g \| _ { H ( k ) } ^ { 2 } } \geq \frac { 1 } { C e ^ { 2 } } \delta ^ { 2 s - d } ,
$$

which completes the proof.

Proof of Theorem 4.3 Let $f \in H ( k )$ . From Proposition 7.4 and the Cauchy–Schwarz inequality it follows that

$$
\begin{array} { r } { | \mathrm { D } _ { x } ^ { \alpha } f ( x ) | \leq \| f \| _ { H ( k ) } \| \mathrm { D } _ { x } ^ { \alpha } k ( \cdot , x ) \| _ { H ( k ) } , } \end{array}\tag{7.12}
$$

for any $\alpha \in \mathbb { N } _ { 0 } ^ { d }$ and $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , where the subscript denotes that diferentiation is with respect to x. Now it follows from the stationarity of k that

$$
\begin{array} { r l } { \| \boldsymbol { \Sigma } _ { x } ^ { \alpha } [ \cdot , x ] \| _ { H ( \mathbb { t } ) } ^ { 2 } = \boldsymbol { \mathrm { D } _ { \varphi } ^ { \alpha } \mathrm { D } _ { \varphi } ^ { \alpha } k ( z , w ) } \| _ { \mathrm { t w = z } } = ( - 1 ) ^ { s / \alpha } \mathrm { D } ^ { 2 \alpha } \widetilde { \boldsymbol { \Phi } } ( 0 ) } \\ & { = \displaystyle \frac { 1 } { ( 2 \pi ) ^ { d / 2 } } \int _ { \mathbb { R } } \omega ^ { 2 \alpha } ( P \Phi ) ( \omega ) \mathrm { d } \omega } \\ & { \leq \displaystyle \frac { C } { ( 2 \pi ) ^ { d / 2 } } \int _ { \mathbb { R } } \omega ^ { 2 \alpha } \mathrm { e s p } ( - \omega \| \omega \| _ { 2 } ) \mathrm { d } \omega } \\ & { \leq \frac { C } { ( 2 \pi ) ^ { d / 2 } } \displaystyle \frac { A } { s = 1 } \int _ { \mathbb { R } } \omega _ { j } ^ { 2 \alpha } \mathrm { e x p } ( - \omega \omega _ { j } ) / \sqrt { d } \mathrm { d } \omega _ { j } } \\ & { \leq \frac { C } { ( 2 \pi ) ^ { d / 2 } } \displaystyle \frac { A } { s = 1 } \int _ { \mathbb { R } } ^ { \infty } \omega _ { j } ^ { 2 \alpha / \alpha } \mathrm { d } \omega } \\ & { = \frac { C } { ( 2 \pi ) ^ { d / 2 } } \left( \frac { \sqrt { d } } { \alpha } \right) ^ { 2 \alpha / \alpha + d } \displaystyle \frac { d } { \int _ { \mathbb { D } } ^ { 1 } \omega _ { j } ^ { 2 \alpha } \mathrm { e x p } ( - | \omega _ { j } | ) \mathrm { d } \omega _ { j } } } \\ & { = \frac { 2 ^ { \ell } C } { ( 2 \pi ) ^ { d / 2 } } \left( \frac { \sqrt { d } } { \alpha } \right) ^ { 2 \alpha / \alpha + d } \displaystyle \frac { d } { \int _ { \mathbb { D } } ^ { 1 } ( 2 \alpha _ { j } ) \mathrm { d } \omega } } \end{array}
$$

A straightforward application of Stirling’s approximation in (7.9) and (7.12) show that f satisfies (4.3) and is thus real analytic. Therefore all functions in $H ( k )$ are real analytic.

Let $B \subset \mathbb { R } ^ { d }$ be an open ball such that $\{ x _ { i } \} _ { i = 1 } ^ { \infty }$ is dense in B. The span of kernel translates for $x _ { i } \in B , W = \operatorname { s p a n } \{ k ^ { x _ { i } } : x _ { i } \in B \}$ , is dense in $H ( k )$ if and only if its orthogonal complement is trivial. Let $f \in H ( k )$ be in the orthogonal complement, which means that $\langle f , k ^ { x _ { i } } \rangle _ { H ( k ) } = f ( x _ { i } ) = 0$ for every $x _ { i } \in B$ . Because $\{ x _ { i } \} _ { i = 1 } ^ { \infty }$ is dense in B and $f$ is real analytic, it follows from Theorem 4.2 that $f = 0$ . Therefore the orthogonal complement of W is trivial and thus W is dense in $H ( k )$ . Let $\boldsymbol { x } \in \mathbb { R } ^ { d }$ and $\varepsilon > 0$ . From W being dense in $H ( k )$ it follows that there are $N \in \mathbb { N } , z _ { 1 } , \dots , z _ { N } \in B$ , and $\mathsf { v } \in \mathbb { R } ^ { N }$ such that

$$
\left\| k ^ { x } - \sum _ { j = 1 } ^ { N } v _ { j } k ^ { z _ { j } } \right\| _ { H ( k ) } \le \frac { \varepsilon } { 2 } .\tag{7.13}
$$

Set $\eta = \varepsilon / ( 2 ( 1 + \| \mathsf { v } \| _ { 1 } ) ) > 0$ . Let $L \in \mathbb { N } .$ . Because $z \mapsto k ^ { z }$ is continuous from $\mathbb { R } ^ { d }$ to $H ( k )$ and the input points are dense in $B ,$ we may choose pairwise disjoint open balls $U _ { 1 } , \dots , U _ { N } \subset B$ and indices $i _ { j , l } \in \mathbb { N }$ for $j \leq N$ and $l \leq L$ such that $x _ { i _ { j , l } } \in U _ { j }$ and

$$
\| k ^ { z _ { j } } - k ^ { x _ { i } _ { j , l } } \| _ { H ( k ) } \leq \eta\tag{7.14}
$$

for all $j$ and l. Let $n = { \mathrm { m a x } } _ { j \leq N , l \leq L } i _ { j , l }$ and define $\mathbf { u } \in \mathbb { R } ^ { n }$ by setting $u _ { i _ { j , l } } = v _ { j } / L$ for $j \leq N$ and $l \leq L$ and $u _ { i } = 0$ otherwise. The triangle inequality gives

$$
\begin{array} { r l } { \left\| k ^ { x } - \displaystyle \sum _ { i = 1 } ^ { n } u _ { i } k ^ { x _ { i } } \right\| _ { H ( k ) } \le \left\| k ^ { x } - \displaystyle \sum _ { j = 1 } ^ { N } v _ { j } k ^ { z _ { j } } \right\| _ { H ( k ) } + \left\| \displaystyle \sum _ { j = 1 } ^ { N } \left( v _ { j } k ^ { z _ { j } } - \displaystyle \sum _ { l = 1 } ^ { L } u _ { i _ { j , l } } k ^ { x _ { i _ { j , l } } } \right) \right\| _ { H ( k ) } } & { } \\ { \leq \left\| k ^ { x } - \displaystyle \sum _ { j = 1 } ^ { N } v _ { j } k ^ { z _ { j } } \right\| _ { H ( k ) } + \displaystyle \sum _ { j = 1 } ^ { N } \frac { | v _ { j } | } { L } \displaystyle \sum _ { l = 1 } ^ { L } \| k ^ { z _ { j } } - k ^ { x _ { i _ { j , l } } } \| _ { H ( k ) } . } & { } \end{array}
$$

From (7.13) and (7.14) it follows that

$$
\left\| k ^ { x } - \sum _ { i = 1 } ^ { n } u _ { i } k ^ { x _ { i } } \right\| _ { H ( k ) } \leq { \frac { \varepsilon } { 2 } } + \eta \| \mathbf { v } \| _ { 1 } \leq \varepsilon .
$$

By Lemma 7.1,

$$
\mathbb { V } _ { \sigma } ( x \mid \mathcal { D } _ { n } ) \le \left\| k ^ { x } - \sum _ { i = 1 } ^ { n } u _ { i } k ^ { x _ { i } } \right\| _ { H ( k ) } ^ { 2 } + \sigma ^ { 2 } \| \mathbf { u } \| _ { 2 } ^ { 2 } \le \varepsilon ^ { 2 } + \sigma ^ { 2 } \sum _ { j = 1 } ^ { N } \sum _ { l = 1 } ^ { L } u _ { i _ { j , l } } ^ { 2 } = \varepsilon ^ { 2 } + \frac { \sigma ^ { 2 } \| \mathbf { v } \| _ { 2 } ^ { 2 } } { L } .
$$

Because L can be chosen arbitrarily large for any given ε and the variance is non-increasing in n by Lemma 7.1, we conclude that for every $\varepsilon > 0$ there is $n _ { \varepsilon } \in \mathbb { N }$ such that $\mathbb { V } _ { \sigma } ( x \mid { \mathcal { D } } _ { n } ) \leq 2 \varepsilon ^ { 2 }$ for every $n \geq n _ { \varepsilon }$ . Therefore $\mathbb { V } _ { \sigma } ( x \mid { \mathcal { D } } _ { n } ) \to 0$ as $n \to \infty$ . Moreover, Dini’s theorem implies that the convergence is uniform on every compact set. ■

Proof of Theorem 4.4 The proof is similar to that of Theorem 4.1. By Lemma 7.3 it sufices to consider the case $\sigma = 0$ . Fix $\boldsymbol { x } \in \mathbb { R } ^ { d }$ and let $\begin{array} { r } { \delta = \delta ( x , X _ { n } ) = \operatorname* { m i n } _ { x _ { i } \in X _ { n } } \| x - x _ { i } \| _ { 2 } } \end{array}$ denote the distance of x to $X _ { n } . ~ \mathrm { { B y } }$ Example 1.4.9 and Theorem 1.6.1 in Rodino (1993), for any $\tau \in ( 0 , 1 )$ there is an infinitely diferentiable bump function $\varphi$ supported on the unit ball such that $\varphi ( 0 ) = 1$ and $( \mathcal { F } \varphi ) ( \omega ) = O ( \exp ( - b \| \omega \| _ { 2 } ^ { \tau } ) )$ for some $b > 0$ . The function $g = \varphi ( ( \cdot - x ) / \delta )$ vanishes on $X _ { n }$ and $g ( x ) = \varphi ( 0 ) = 1$ . It immediately follows from the spectral characterisation (7.2) that $g \in H ( k )$ if $k _ { \eta }$ is a Gevrey kernel of order $\eta < \tau$ . Just as in the proof of Theorem 4.1 we obtain

$$
\mathbb { V } ( x \mid \mathcal { D } _ { n } ) \geq \frac { 1 } { \| g \| _ { H ( k _ { \eta } ) } ^ { 2 } }
$$

from (7.4). It remains to estimate $\| g \| _ { H ( k _ { \eta } ) }$ . There is a positive constant $C _ { 1 }$ independent of $\delta$ such that

$$
\begin{array} { r l r } {  { \Vert g \Vert _ { H ( k _ { \eta } ) } ^ { 2 } = \frac { 1 } { ( 2 \pi ) ^ { d / 2 } } \int _ { \mathbb { R } ^ { d } } \frac { \vert ( \mathcal { F } g ) ( \omega ) \vert ^ { 2 } } { ( \mathcal { F } \Phi _ { \eta } ) ( \omega ) } \mathrm { d } \omega = \frac { \delta ^ { 2 d } } { C _ { \eta } ( 2 \pi ) ^ { d / 2 } } \int _ { \mathbb { R } ^ { d } } \vert ( \mathcal { F } \varphi ) ( \delta \omega ) \vert ^ { 2 } \exp ( \rho \Vert \omega \Vert _ { 2 } ^ { \eta } ) \mathrm { d } \omega } } \\ & { } & { = \frac { \delta ^ { d } } { C _ { \eta } ( 2 \pi ) ^ { d / 2 } } \int _ { \mathbb { R } ^ { d } } \vert ( \mathcal { F } \varphi ) ( \omega ) \vert ^ { 2 } \exp ( \rho \delta ^ { - \eta } \Vert \omega \Vert _ { 2 } ^ { \eta } ) \mathrm { d } \omega } \\ & { } & { \le \frac { C _ { 1 } } { C _ { \eta } ( 2 \pi ) ^ { d / 2 } } \delta ^ { d } \int _ { \mathbb { R } ^ { d } } \exp ( \rho \delta ^ { - \eta } \Vert \omega \Vert _ { 2 } ^ { \eta } - 2 b \Vert \omega \Vert _ { 2 } ^ { \tau } ) \mathrm { d } \omega . } \end{array}
$$

The exponent in the integral has the maximum

$$
2 b ( \tau / \eta - 1 ) \left( \frac { \eta \rho } { 2 \tau b } \right) ^ { 1 / ( 1 - \eta / \tau ) } \delta ^ { - 1 / ( 1 / \eta - 1 / \tau ) } = : \theta \delta ^ { - 1 / ( 1 / \eta - 1 / \tau ) }
$$

and is less than $- b \| \omega \| _ { 2 } ^ { \tau }$ when $\| \omega \| _ { 2 } > r ( \delta ) : = ( \rho b ^ { - 1 } ) ^ { 1 / ( \tau - \eta ) } \delta ^ { - 1 / ( \tau / \eta - 1 ) }$ . Therefore there are positive constants $C _ { 2 }$ and $C _ { 3 }$ that do not depend on $\delta$ such that

$$
\begin{array} { r l } & { \| g \| _ { H ( k _ { \eta } ) } ^ { 2 } \le C _ { 2 } \delta ^ { d } \bigg ( \int _ { \| \omega \| _ { 2 } \le r ( \delta ) } \exp ( \theta \delta ^ { - 1 / ( 1 / \eta - 1 / \tau ) } ) \mathrm { d } \omega + \int _ { \| \omega \| _ { 2 } > r ( \delta ) } \exp ( - b \| \omega \| _ { 2 } ^ { \tau } ) \mathrm { d } \omega \bigg ) } \\ & { \qquad \le C _ { 3 } \delta ^ { ( 1 / \eta - 2 / \tau ) d / ( 1 / \eta - 1 / \tau ) } \exp ( \theta \delta ^ { - 1 / ( 1 / \eta - 1 / \tau ) } ) } \end{array}
$$

when $\delta$ is suficiently small. Here we used the fact that the volume of an r-ball in $\mathbb { R } ^ { d }$ is proportional to $r ^ { d }$ . The claim now follows.

## Acknowledgements

TK was supported by the Research Council of Finland grants 338567 (“Scalable, adaptive and reliable probabilistic integration”), 359183 (“Flagship of Advanced Mathematics for Sensing, Imaging and Modelling”), and 368086 (“Inference and approximation under misspecification”). TK also acknowledges the research environment provided by ELLIS Institute Finland. CJO was supported by a Philip Leverhulme Prize (PLP-2023-004). The origins of this article are in the positive response that TK received to talks given on the topic at the Prob Num 2022 workshop in London in March 2022 and at the ELISE Theory Workshop on Machine Learning Fundamentals at EURECOM in September 2022. We are grateful to Anatoly Zhigljavsky for discussions that helped to kindle our interest in inverse multiquadrics, to Luc Pronzato for introducing us to literature on the no-empty-ball property, and to Vesa Kaarnioja for ample discussion on Gevrey spaces and kernels.

## References

R. Ababou, A. C. Bagtzoglou, and E. F. Wood. On the condition number of covariance matrices in kriging, estimation, and simulation of random fields. Mathematical Geology, 26(1):99–133, 1994.

M. Belkin. Approximation beats concentration? An approximation view on inference with smooth radial kernels. In Proceedings of the 31st Conference on Learning Theory, volume 75 of Proceedings of Machine Learning Research, pages 1348–1361, 2018.

A. Berlinet and C. Thomas-Agnan. Reproducing Kernel Hilbert Spaces in Probability and Statistics. Springer, 2004.

A. Chernih, I. H. Sloan, and R. S. Womersley. Wendland functions with increasing smoothness converge to a Gaussian. Advances in Computational Mathematics, 40(1):185–200, 2014.

P. Diamond and M. Armstrong. Robustness of variograms and conditioning of kriging matrices. Journal of the International Association for Mathematical Geology, 16(8): 809–822, 1984.

B. Diederichs and A. Iske. Improved estimates for condition numbers of radial basis function interpolation matrices. Journal of Approximation Theory, 238:38–51, 2019.

M. Eberts and I. Steinwart. Optimal regression rates for SVMs using Gaussian kernels. Electronic Journal of Statistics, 7:1–42, 2013.

J. R. Gardner, G. Pleiss, D. Bindel, K. Q. Weinberger, and A. G. Wilson. GPyTorch: Blackbox matrix-matrix Gaussian process inference with GPU acceleration. In Advances in Neural Information Processing Systems, volume 31, 2018.

T. M. Garoni and N. E. Frankel. L´evy flights: Exact results and asymptotics beyond all orders. Journal of Mathematical Physics, 43:2670–2689, 2002.

GPy. GPy: A Gaussian process framework in Python. http://github.com/SheffieldML/ GPy, since 2012.

R. B. Gramacy. Surrogates: Gaussian Process Modeling, Design, and Optimization for the Applied Sciences. CRC Press, 2020.

R. B. Gramacy and H. K. H. Lee. Cases for the nugget in modeling computer experiments. Statistics and Computing, 22:713–722, 2012.

M. Gu, X. Wang, and J. O. Berger. Robust Gaussian stochastic process emulation. The Annals of Statistics, 46(6A):3038–3066, 2018.

M. S. Handcock and M. L. Stein. A Bayesian analysis of kriging. Technometrics, 35(4): 403–410, 1993.

P. Hennig, M. A. Osborne, and H. P. Kersting. Probabilistic Numerics. Cambridge University Press, 2022.

E. Hewitt and K. Stromberg. Real and Abstract Analysis. Springer-Verlag, 1965.

M. Kanagawa, P. Hennig, D. Sejdinovic, and B. K. Sriperumbudur. Gaussian processes and reproducing kernels: Connections and equivalences. arXiv:2506.17366v1, 2025.

T. Karvonen. Bumps and polynomials in RKHSs of translation-invariant kernels. arXiv:2102.10628v2, 2026.

T. Karvonen and F. Bachoc. Scale estimation and rate-unbiasedness for Gaussian processes under smoothness misspecification. arXiv:2110.02810v3, 2025.

T. Karvonen and Y. Suzuki. Approximation in Hilbert spaces of the Gaussian and related analytic kernels. IMA Journal of Numerical Analysis, 46(4):2344–2376, 2026.

T. Karvonen, G. Wynne, F. Tronarp, C. J. Oates, and S. S¨arkk¨a. Maximum likelihood estimation and uncertainty quantification for Gaussian process approximation of deterministic functions. SIAM/ASA Journal on Uncertainty Quantification, 8(3):926–958, 2020.

G. S. Kimeldorf and G. Wahba. A correspondence between Bayesian estimation on stochastic processes and smoothing by splines. The Annals of Mathematical Statistics, 41(2):495–502, 1970.

A. Kolmogorov. Interpolation and extrapolation of stationary random sequences. Izvestiya Akademii Nauk SSSR. Seriya Matematicheskaya, 5(1):3–14, 1941. In Russian.

S. G. Krantz and H. R. Parks. A Primer of Real Analytic Functions. Birkh¨auser, 2nd edition, 2002.

M. L´azaro-Gredilla, J. Quinonero-Candela, C. E. Rasmussen, and A. R. Figueiras-Vidal. Sparse spectrum Gaussian process regression. Journal of Machine Learning Research, 11: 1865–1881, 2010.

J. P. Lewis. Generalized stochastic subdivision. ACM Transactions on Graphics, 6(3): 167–190, 1987.

J. Lloyd, D. Duvenaud, R. Grosse, J. Tenenbaum, and Z. Ghahramani. Automatic construction and natural-language description of nonparametric regression models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 28, 2014.

C. A. Micchelli, Y. Xu, and H. Zhang. Universal kernels. Journal of Machine Learning Research, 7:2651–2667, 2006.

H. Q. Minh. Some properties of Gaussian reproducing kernel Hilbert spaces and their implications for function approximation and learning theory. Constructive Approximation, 32(2):307–338, 2010.

B. S. Mityagin. The zero set of a real analytic function. Mathematical Notes, 107:529–530, 2020.

K. Muandet, K. Fukumizu, B. Sriperumbudur, and B. Sch¨olkopf. Kernel mean embedding of distributions: A review and beyond. Foundations and Trends® in Machine Learning, 10(1–2):1–141, 2017.

M. Naslidnyk, M. Kanagawa, T. Karvonen, and M. Mahsereci. Comparing scale parameter estimators for Gaussian process interpolation with the Brownian motion prior: Leaveone-out cross validation and maximum likelihood. SIAM/ASA Journal on Uncertainty Quantification, 2025. To appear.

R. M. Neal. Regression and classification using Gaussian process priors. In J. M. Bernardo, J. O. Berger, A. P. Dawid, and A. F. M. Smith, editors, Bayesian Statistics 6, pages 475–502, 1998.

E. Novak, M. Ullrich, H. Wo´zniakowski, and S. Zhang. Reproducing kernels of Sobolev spaces on R<sup>d</sup> and applications to embedding constants and tractability. Analysis and Applications, 16(5):693–715, 2018.

V. I. Paulsen and M. Raghupathi. An Introduction to the Theory of Reproducing Kernel Hilbert Spaces. Number 152 in Cambridge Studies in Advanced Mathematics. Cambridge University Press, 2016.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.

C.-Y. Peng and C. F. J. Wu. On the choice of nugget in kriging modeling for deterministic computer experiments. Journal of Computational and Graphical Statistics, 23(1):151–168, 2014.

S. J. Petit, J. Bect, and E. Vazquez. Relaxed Gaussian process interpolation: a goal-oriented approach to Bayesian optimization. arXiv:2206.03034v2, 2022.

R. B. Platte. How fast do radial basis function interpolants of analytic functions converge? IMA Journal of Numerical Analysis, 31(4):1578–1597, 2011.

A. Poltoratski. Spectral gaps for sets and measures. Acta Mathematica, 208(1):151–209, 2012.

D. Posa. Conditioning of the stationary kriging matrices for some well-known covariance models. Mathematical Geology, 21(7):755–765, 1989.

C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for Machine Learning. Adaptive Computation and Machine Learning. MIT Press, 2006.

M. Reed and B. Simon. Methods of Modern Mathematical Physics. II: Fourier Analysis, Self-Adjointness. Elsevier, 1975.

S. Remes, M. Heinonen, and S. Kaski. Non-stationary spectral kernels. Advances in Neural Information Processing Systems, 30, 2017.

C. Rieger and B. Zwicknagl. Sampling inequalities for infinitely smooth functions, with applications to interpolation and machine learning. Advances in Computational Mathematics, 32:103–129, 2010.

H. Robbins. A remark on Stirling’s formula. The American Mathematical Monthly, 62(1): 26–29, 1955.

L. Rodino. Linear Partial Diferential Operators in Gevrey Spaces. World Scientific, 1993.

J. Sacks, W. J. Welch, T. J. Mitchell, and H. P. Wynn. Design and analysis of computer experiments. Statistical Science, 4(4):409–435, 1989.

R. Schaback. Error estimates and condition numbers for radial basis function interpolation. Advances in Computational Mathematics, 3(3):251–264, 1995.

R. Schaback and H. Wendland. Kernel techniques: From machine learning to meshless methods. Acta Numerica, 15:543–639, 2006.

A. Solin and S. S¨arkk¨a. Hilbert space methods for reduced-rank Gaussian process regression. Statistics and Computing, 30(2):419–446, 2020.

M. L. Stein. Interpolation of Spatial Data: Some Theory for Kriging. Springer Series in Statistics. Springer, 1999.

M. L. Stein. The screening efect in Kriging. The Annals of Statistics, 30(1):298–323, 2002.

M. L. Stein. When does the screening efect hold? The Annals of Statistics, 39(6):2795–2819, 2011.

M. L. Stein. When does the screening efect not hold? Spatial Statistics, 11:65–80, 2015.

I. Steinwart and A. Christmann. Support Vector Machines. Information Science and Statistics. Springer, 2008.

H.-W. Sun and D.-X. Zhou. Reproducing kernel Hilbert spaces associated with analytic translation-invariant Mercer kernels. Journal of Fourier Analysis and Applications, 14(1): 89–101, 2008.

B. Szab´o, A. W. van der Vaart, and J. H. van Zanten. Frequentist coverage of adaptive nonparametric Bayesian credible sets. The Annals of Statistics, 43(4):1391–1428, 2015.

F. Tronarp and T. Karvonen. Orthonormal expansions for translation-invariant kernels. Journal of Approximation Theory, 302:106055, 2024.

E. Vazquez and J. Bect. Convergence properties of the expected improvement algorithm with fixed mean and covariance functions. Journal of Statistical Planning and Inference, 140(11):3088–3095, 2010a.

E. Vazquez and J. Bect. Pointwise consistency of the kriging predictor with known mean and covariance functions. In mODa 9—Advances in Model-Oriented Design and Analysis, pages 221–228. Physica-Verlag HD, 2010b.

H. Wendland. Scattered Data Approximation. Number 17 in Cambridge Monographs on Applied and Computational Mathematics. Cambridge University Press, 2005.

N. Wiener. Extrapolation, Interpolation, and Smoothing of Stationary Time Series. MIT Press, 1949.

C. K. I. Williams and C. E. Rasmussen. Gaussian processes for regression. In Advances in Neural Information Processing Systems, volume 8, pages 514–520. MIT Press, 1996.

A. G. Wilson, Z. Hu, R. Salakhutdinov, and E. P. Xing. Deep kernel learning. In Artificial Intelligence and Statistics, volume 51 of Proceedings of Machine Learning Research, pages 370–378, 2016.

D. Yarotsky. Examples of inconsistency in optimization by expected improvement. Journal of Global Optimization, 56:1773–1790, 2013a.

D. Yarotsky. Univariate interpolation by exponential functions and Gaussian RBFs for generic sets of nodes. Journal of Approximation Theory, 166:163–175, 2013b.

V. M. Zolotarev. One-Dimensional Stable Distributions. Number 65 in Translations of Mathematical Monographs. American Mathematical Society, 1986.