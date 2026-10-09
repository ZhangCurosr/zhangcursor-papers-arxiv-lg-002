# Diffusion Removes Langevin’s Conditioning Dependence: A Sharp Gaussian Analysis

Adam Perbost, Francis Bach, Pierre Marion

Inria, Ecole normale sup <sup>´</sup> erieure – PSL Research University´

Paris, France

{adam.perbost,francis.bach,pierre.marion}@inria.fr

October 9, 2026

## Abstract

Despite their empirical success, why diffusion models overcome the bottlenecks of classical score-based samplers remains unclear. In this work, we leverage Gaussian distributions to isolate this phenomenon. We establish 2-Wasserstein convergence bounds for optimized hyperparameters, showing that diffusion processes <sub>achieve a sampling error of O(</sub>√<sub>dλ log N/N), where d is the dimension, N the number of sampling steps, and</sub> $\lambda _ { \mathrm { m a x } }$ the largest eigenvalue of the target covariance matrix. Unadjusted and underdamped Langevin dynamics suffer from an additional $\sqrt { \kappa }$ factor, where κ is the condition number. These rates follow from spectral bounds which are sharp: we confirm them via matching first-order asymptotics as $N  \infty$ . Our analysis provides a rigorous characterization, in the Gaussian setting, of how time-dependent score trajectories remove conditionnumber dependence during sampling. By contrast, in the learning phase, we show that estimating the unnoised score by gradient descent leads to essentially the same estimator as estimating a noisy score, which suggests that the benefits of noising do not come from the learning phase.

## 1 Introduction

Since their introduction, diffusion models [Sohl-Dickstein et al., 2015, Song and Ermon, 2019, Ho et al., 2020, Song et al., 2021] have achieved state-of-the-art performance across a wide range of domains [Dhariwal and Nichol, 2021, Kong et al., 2021, Hoogeboom et al., 2022, Watson et al., 2023]. Training diffusion models requires estimating the gradient of the log-density (the so-called score) of a noisy version of the target. Then samples are produced by iterative denoising through a formula involving this score. Taking a step back, iterative score-based sampling strategies long predate modern diffusion models, as epitomized by the celebrated Langevin algorithm [Pavliotis, 2014, Roberts and Tweedie, 1996] and its many variants such as underdamped Langevin [Leimkuhler and Matthews, 2015]. The key factors explaining their success compared to those more classical algorithms are still unclear: what exact role does the noising process in diffusion models play, and how does it interact with the learning and sampling phases?

Our primary goal in this work is to prove a separation in sampling efficiency between diffusion models and Langevin dynamics in terms of their dependence on the condition number. This sheds light on the practical effectiveness of diffusion models on ill-conditioned distributions such as natural images [Guth et al., 2022, Field, 1987]. To do so, we consider the class of (centered) Gaussian targets. Naturally, sampling from a Gaussian distribution is not our primary concern here. Rather, we use this analytically tractable setting to derive sharp convergence rates accompanied by matching asymptotics, in order to highlight how diffusion models overcome limitations of other score-based sampling methods. As discussed below, deriving tight condition-number dependence for more general target classes such as log-concave distributions is a long-standing open problem [Cheng et al., 2018, Dalalyan and Riou-Durand, 2020], which is outside the scope of the present paper. Additionally, we focus for most of the paper on the sampling phase, comparing the sampling convergence rates of diffusion, unadjusted Langevin algorithm (ULA) and underdamped Langevin dynamics (ULD). Assuming access to the true score yields a clean analysis and the separation between diffusion and Langevin, hence for simplicity we focus on this case. The connection with the estimation of the score in the learning phase is discussed in the latter part of our work.

Table 1: Convergence rates and ε-complexities for the Gaussian target. Rates are the orders of $W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } )$ as $N \to \infty$ for fixed Σ (Theorems 1 and 3). The ε-complexity is the iteration budget N needed to achieve $W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \varepsilon$ , where O<sup>˜</sup> hides logarithmic factors. Diffusion models remove both the conditioning factor $\sqrt { \kappa }$ of Langevin samplers and the burn-in of order κ.
<table><tr><td></td><td>Algorithm Asymptotic equivalent ε-complexity</td></tr><tr><td>ULA</td><td> $\begin{array} { r } { \Theta \left( \lambda _ { \operatorname* { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \frac { \log N } { N } \right) } \end{array}$   $\begin{array} { r } { \tilde { O } \big ( \kappa + \frac { \sqrt { d \kappa \lambda _ { \mathrm { m a x } } } } { \varepsilon } \big ) } \end{array}$ </td></tr><tr><td>ULD</td><td> $\begin{array} { r } { \Theta \left( \lambda _ { \operatorname* { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \frac { \log N } { N } \right) } \end{array}$   $\begin{array} { r } { \tilde { O } \big ( \kappa + \frac { \sqrt { d \kappa \lambda _ { \mathrm { m a x } } } } { \varepsilon } \big ) } \end{array}$ </td></tr><tr><td>Diffusion</td><td> $\Theta \biggl ( \sqrt { \mathrm { T r } \Sigma } { \frac { \log N } { N } } \biggr )$   $\begin{array} { r } { \dot { \tilde { O } } \left( \frac { \sqrt { d \lambda _ { \operatorname* { m a x } } } } { \varepsilon } \right) } \end{array}$ </td></tr></table>

Contributions. We establish upper bounds on the 2-Wasserstein error between the target distribution and the distribution sampled from the diffusion, ULA and ULD. Convergence rates are derived in Section 2 by optimizing these bounds with respect to hyperparameters (see Table 1 for summary). The bounds for diffusion and for Langevin differ by a factor proportional to the square root of the condition number. Thanks to explicit computations of the Wasserstein errors, we confirm these rates via matching first-order asymptotics. By contrast, in the learning phase, we show in Section 3 that estimating the unnoised score by gradient descent leads to essentially the same estimator as estimating a noisy score, the two differing only in how they regularize the directions unseen in the data, which suggests that the benefits of noising do not come from the learning phase.

Related work. For ULA, a similar analysis of the two error terms was already performed, for example, by Durmus et al. [2019], Wibisono [2018] and specifically in the Gaussian case by Pedregosa [2023]. Building upon these works, we perform hyperparameter optimization and compute the asymptotic equivalent (rather than upper bounds). Many works have studied ULD in the log-concave case, see, e.g., Cheng et al. [2018], Dalalyan and Riou-Durand [2020], Leimkuhler et al. [2024], Kim et al. [2026] for studies of the stochastic Euler scheme and Euler–Maruyama (EM) integrators. This line of work yields loose dependence on the condition number due to the bias term induced by non-vanishing step-sizes (see, e.g., Eq. (11) in Dalalyan and Riou-Durand [2020] with a $\kappa ^ { 3 / 2 }$ in the complexity bound compared to our $\sqrt { \kappa } )$ . A line of work has proved improved condition dependence for Gaussian targets [Shen and Lee, 2019, Apers et al., 2024, Gouraud et al., 2025], albeit with more complex integrators (splitting, random step-sizes, or Metropolis corrections). To our knowledge, the tight study of the bias of ULD for a Gaussian target and the EM integrator is a novel contribution

For diffusion models, Pierret and Galerne [2025] derive and numerically analyze error terms for Gaussian targets, but do not analytically derive optimal hyperparameters, discuss the dependence on the condition number, nor derive asymptotic equivalents. Closer are the works by Hurault et al. [2025] and Hurault et al. [2026], which derive Wasserstein bounds for Gaussian data and discuss optimal hyperparameter choices depending on the covariance spectrum. A key difference with our work is that their analysis holds asymptotically for small step-sizes, whereas we make a fully non-asymptotic analysis (albeit in less generality), which enables us to capture the interaction between the step-size and other parameters, and to derive optimal step complexity with respect to our bounds. Wasserstein bounds beyond the Gaussian case are given for instance by Strasman et al. [2025], Beyler and Bach [2025].

## 2 Sampling: diffusion removes the condition-number dependence

## 2.1 Problem setup

We denote by P the target d-dimensional centered Gaussian distribution $\mathcal { N } ( 0 , \Sigma )$ . Let $( \lambda _ { i } ) _ { 1 \leq i \leq d }$ denote the eigenvalues of the positive-definite covariance matrix $\Sigma ,$ and $\lambda _ { \operatorname* { m i n } }$ and $\lambda _ { \mathrm { m a x } }$ be respectively the smallest and largest eigenvalues. The ratio

$$
\kappa : = \frac { \lambda _ { \operatorname* { m a x } } } { \lambda _ { \operatorname* { m i n } } }
$$

is the condition number of $\Sigma ,$ which measures the degree of anisotropy of P. Ill-conditioning is a classical source of difficulty for Langevin-based sampling methods, as their convergence rates typically deteriorate with $\kappa$ [Cheng et al., 2018, Dalalyan and Riou-Durand, 2020].

Let $p$ denote the Lebesgue density of P. We define the score s of the target distribution as

$$
s : x \mapsto \nabla \log p ( x ) = - \Sigma ^ { - 1 } x .
$$

We assume that the score is known exactly in this section; the learning problem is studied separately in Section 3.

We measure sampling error in the 2-Wasserstein distance, denoted by $W _ { 2 }$ . For two centered Gaussians whose covariance matrices $\Sigma _ { 1 }$ and $\Sigma _ { 2 }$ commute, the distance reduces to the Frechet distance´ $\| \boldsymbol { \Sigma } _ { 1 } ^ { 1 / 2 } - \boldsymbol { \Sigma } _ { 2 } ^ { 1 / 2 } \| _ { F }$ , with $\left\| \cdot \right\| _ { F }$ the Frobenius norm [Gelbrich, 1990]. For each sampling scheme, we denote by $\mathbb { P } _ { k }$ the law of its k-th iterate.

Time-homogeneous Langevin samplers. We first consider the Langevin process associated with $\mathbb { P } ,$

$$
\mathrm { d } X _ { t } = s ( X _ { t } ) \mathrm { d } t + \sqrt { 2 } \mathrm { d } W _ { t } , t \in \mathbb { R } _ { + } , X _ { 0 } = 0 ,\tag{1}
$$

where $( W _ { t } )$ is a standard Brownian motion.

The unadjusted Langevin algorithm (ULA) [Dalalyan, 2017] is simply the Euler–Maruyama discretization of the SDE in (1). Given a step-size $\eta ,$ an iteration budget $N$ , and a sequence $\left( \xi _ { k } \right)$ of independent standard Gaussian vectors in $\mathbb { R } ^ { d }$ , it is defined by

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { X _ { 0 } = 0 , } \\ { \forall k \in \{ 0 , \ldots , N - 1 \} , X _ { k + 1 } = X _ { k } + \eta s ( X _ { k } ) + \sqrt { 2 \eta } \xi _ { k } . } \end{array} \right. } \end{array}\tag{2}
$$

We next consider underdamped Langevin dynamics (ULD) [Cheng et al., 2018], which introduces an auxiliary velocity variable into the Langevin dynamics with the aim of accelerating convergence:

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \mathrm { d } X _ { t } = V _ { t } \mathrm { d } t , } \\ { \mathrm { d } V _ { t } = - \gamma V _ { t } \mathrm { d } t + s ( X _ { t } ) \mathrm { d } t + \sqrt { 2 \gamma } \mathrm { d } W _ { t } , } \end{array} \right. \ t \in \mathbb { R } _ { + } , } \end{array}\tag{3}
$$

with initialization $X _ { 0 } = V _ { 0 } = 0$ , where $\gamma > 0$ is the friction coefficient. We discretize these dynamics using the Euler–Maruyama scheme, yielding with the same notation as above

$$
\left\{ \begin{array} { l l } { X _ { k + 1 } = X _ { k } + \eta V _ { k } , } \\ { V _ { k + 1 } = ( 1 - \eta \gamma ) V _ { k } + \eta s ( X _ { k } ) + \sqrt { 2 \eta \gamma } \xi _ { k } . } \end{array} \right.\tag{4}
$$

Time-inhomogeneous diffusion sampler. Diffusion-based sampling follows a different principle [Sohl-Dickstein et al., 2015, Song et al., 2021]: rather than sampling directly with the score of the target distribution, it constructs a family of perturbed distributions and uses a time-dependent score along this family. Many conventions exist; we study the following one because it leads to telescoping cancellations in the computations, greatly simplifying our analysis (see Appendix C.2.1). Concretely, for a general (centered) target $\mathbb { Q } ,$ consider the forward noising process

$$
\left\{ \begin{array} { l l } { \mathrm { d } X _ { t } = \mathrm { d } W _ { t } , t \in [ 0 , T ] , } \\ { X _ { 0 } = X \sim \mathbb { Q } , } \end{array} \right.
$$

whose marginal $\mathbb { Q } _ { t }$ at time t has Lebesgue density $q _ { t }$ and score $s _ { t } : = \nabla$ log $q _ { t }$ . The forward diffusion gradually smooths the target distribution, and $T$ is chosen large enough so that $\mathbb { Q } _ { T } \approx \mathcal { N } ( 0 , T \mathrm { I } )$ , a distribution that is straightforward to sample from. Sampling proceeds by reversing this transformation, i.e., by simulating the reversetime SDE [Anderson, 1982]

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \mathrm { d } Y _ { t } = s _ { T - t } ( Y _ { t } ) \mathrm { d } t + \mathrm { d } W _ { t } , t \in [ 0 , T - \delta ] , } \\ { Y _ { 0 } \sim \mathcal { N } ( 0 , T \mathrm { I } ) , } \end{array} \right. } \end{array}\tag{5}
$$

where ${ \mathcal { N } } ( 0 , T \operatorname { I } )$ replaces the intractable true marginal $\mathbb { Q } _ { T }$ and $\delta > 0$ is an early-stopping threshold, standard in the analysis of diffusion models, where it prevents a possible blow-up of the score as $t  0 [ \mathrm { D e }$ Bortoli, 2022, Benton et al., 2024]. In our Gaussian setting $\mathbb { Q } = \mathbb { P }$ , all these quantities are explicit: the marginal at time t is $\mathbb { Q } _ { t } = \mathcal { N } ( 0 , C _ { t } )$ with $C _ { t } = \Sigma + t \mathrm { I }$ , and

$$
s _ { t } : x \mapsto - C _ { t } ^ { - 1 } x ,
$$

whose Lipschitz constant $( \lambda _ { \operatorname* { m i n } } + t ) ^ { - 1 }$ is bounded by $( \lambda _ { \operatorname* { m i n } } + \delta ) ^ { - 1 }$ on $[ \delta , T ]$

Since information about the target is concentrated near $t = T - \delta$ , we discretize (5) using a decreasing sequence of step-sizes $( \eta _ { k } )$ instead of a constant η. We fix a decreasing grid $T = \tau _ { 0 } > \tau _ { 1 } > \cdots > \tau _ { N } = \delta$ defining $\tau _ { k } = \tau _ { k - 1 } - \eta _ { k }$ for $k \in \{ 1 , \ldots , N \}$ . The diffusion algorithm we study is then

$$
\begin{array} { r l } & { \left\{ Y _ { 0 } \sim { \mathcal N } ( 0 , T \mathrm { I } ) , \right. } \\ & { \left. \forall k = 1 , \dots , N , \ Y _ { k } = Y _ { k - 1 } + \eta _ { k } s _ { \tau _ { k - 1 } } ( Y _ { k - 1 } ) + \sqrt { \eta _ { k } } \xi _ { k } . \right. } \end{array}\tag{6}
$$

The main difference between diffusion and Langevin lies in the use of a time-dependent score $s _ { t }$ in the former. Langevin methods only use the target score s, and therefore operate under a fixed geometry governed by $\kappa .$ This leads, as we will show in Theorem 1, to a penalizing $\sqrt { \kappa }$ factor in the convergence rate. In contrast, diffusion progressively modifies the geometry of the score along the sampling trajectory. Indeed, note that the condition number associated with $s _ { t }$ is

$$
\kappa _ { t } = \frac { t + \lambda _ { \operatorname* { m a x } } } { t + \lambda _ { \operatorname* { m i n } } } \underset { t \to \infty } { \longrightarrow } 1 ,
$$

decreasing from $\kappa _ { 0 } = \kappa$ to 1. Sampling therefore begins in a near-isotropic regime, and only gradually recovers the original conditioning of the target as the trajectory approaches $t = 0$ . This intuition is formalized in Theorem 3, showing that the sampling complexity of diffusion is indeed independent of $\kappa .$

## 2.2 Langevin samplers: the condition-number dependence

Table 1 summarizes our guarantees, with hyperparameters optimized for each method, and Figures 1 and 2 illustrate them. Proofs are given in the appendices. We first consider ULA and ULD. The ULA bound builds on the Gaussian analysis of Pedregosa [2023]. The convergence rates for ULA and ULD differ only by numerical constants.

![](images/5d2250a6d663572a3396db9c37a06023e23b8b1592a750d6d033e7769624af1b.jpg)  
Figure 1: Sampling error $W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } )$ as a function of the iteration budget N (log–log scale) on an ill-conditioned Gaussian target: $d \ = \ 1 0 0 0 .$ , log-uniform spectrum $\lambda _ { i } ~ = ~ \kappa ^ { - ( i - 1 ) / ( d - 1 ) }$ with $\kappa ~ = ~ 1 0 ^ { 6 }$ . For this spectrum, $\lambda _ { \mathrm { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) / \mathrm { T r } \Sigma } = \sqrt { \kappa }$ . Each point is a separate run: for every N, the sampler is tuned as in Theorems 1 and 3, run for N steps, and the error is evaluated in closed form. Dashed lines: asymptotic equivalents of the two theorems, drawn from $N _ { \star }$ onwards for the Langevin samplers. Segments $N < N _ { \star } \colon$ the step-size of Theorem 1 is inadmissible (Remark 2) and the largest admissible one is used instead, $\eta = \lambda _ { \mathrm { m i n } }$ for $\mathrm { U L A }$ and $\eta = 1 / \gamma$ for ULD; see the discussion after Remark 2

Theorem 1 (Langevin samplers). Let $N \geq 2$ be the iteration budget, and set the step-size η (and thefriction $\gamma f o r$ $U L D ) a s$

$$
\begin{array} { r } { \bullet U L A \colon \eta = \frac { 1 } { 2 } \lambda _ { \operatorname* { m a x } } \frac { \log N } { N } , p r o { \nu } i d e d N \geq \frac { 1 } { 2 } \kappa \log N ; } \end{array}
$$

$$
\begin{array} { r } { \bullet \ U L D \colon \eta = \sqrt { 2 \lambda _ { \operatorname* { m a x } } \kappa } \frac { \log N } { N } a n d \gamma = \sqrt { 8 / \lambda _ { \operatorname* { m i n } } } , p r o \nu i d e d N \geq 4 \kappa \log N . } \end{array}
$$

Then

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \frac { \lambda _ { \operatorname* { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } { N } \left( C _ { 1 } \log N + C _ { 2 } \right) ,
$$

with $\begin{array} { r } { C _ { 1 } = \frac { 1 } { 4 } , C _ { 2 } = \sqrt { 2 } } \end{array}$ for ULA and $\begin{array} { r } { C _ { 1 } = \frac { 3 } { 7 } , C _ { 2 } = 8 } \end{array}$ for ULD. Furthermore,forfixed Σ, these bounds are tight up to a constantfactor:

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \underset { N  \infty } { \sim } C _ { 3 } \lambda _ { \operatorname* { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \frac { \log N } { N } ,
$$

with $\begin{array} { r } { C _ { 3 } = \frac { 1 } { 8 } f o r } \end{array}$ ULA and $\begin{array} { r } { C _ { 3 } = \frac { 1 } { 4 } f o r \ : U L D } \end{array}$

Both Langevin samplers deteriorate with the condition number through a $\sqrt { \kappa }$ factor, which can be made explicit by noting that $\begin{array} { r } { \lambda _ { \operatorname* { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } = \sqrt { \lambda _ { \operatorname* { m a x } } } \sqrt { \sum _ { i = 1 } ^ { d } \frac { \lambda _ { \operatorname* { m a x } } } { \lambda _ { i } } \in [ \sqrt { \lambda _ { \operatorname* { m a x } } \kappa } , \sqrt { \lambda _ { \operatorname* { m a x } } d \kappa } ] } } \end{array}$ . The proof of this theorem, given in Appendix B, exploits the Gaussian structure of the target, reducing the analysis to the eigenvalues of the covariance matrix, for which a closed-form expression can be obtained. This yields a Wasserstein error bound consisting of two terms: a discretization error, arising from the discretization of the continuous-time Langevin dynamics, and an early-stopping error, resulting from terminating the algorithm at time $N \eta$ rather than letting it run indefinitely. The resulting hyperparameter choices and convergence rates therefore reflect a trade-off between these two errors.

![](images/7fa746971ca69c42d55b8222e12d8dd25d6f604de647eddc45c52f1c00372bb0.jpg)  
(a) Error ratio versus $\kappa _ { \mathrm { e f f } }$ $N = 1 0 ^ { 1 2 }$

![](images/acd101ae554bb8aaadb4edbbefb5f6d2940129108a365da7dc13b3ace81ca836.jpg)  
(b) ε-complexity versus $\kappa , \varepsilon = 1 0 ^ { - 3 } \sqrt { \mathrm { T r } \Sigma }$  
Figure 2: Tightness of the theory in dimension $d = 1 0 0 0$ . (a) Ratio $W _ { 2 } ^ { \mathrm { L a n g e v i n } } / W _ { 2 } ^ { \mathrm { d i f f u s i o n } }$ at budget $N = 1 0 ^ { 1 2 }$ , as a function of $\kappa _ { \mathrm { e f f } } : = \lambda _ { \mathrm { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) / \mathrm { T r } \Sigma }$ , for four spectral families with $\lambda _ { \operatorname* { m a x } } = 1$ and $\kappa \in \{ 1 0 , 1 0 ^ { 2 } , . . . , 1 0 ^ { 8 } \}$ “log-uniform” $( \lambda _ { i } = \kappa ^ { - ( i \dot { - } 1 ) / ( d - 1 ) } )$ ; “power law” $( \lambda _ { i } = i ^ { - a }$ with $a = \log \kappa / \log d ) ;$ ; “two-block $1 / 2 ^ { , , }$ (half of the eigenvalues equal to 1 and half to $1 / \kappa )$ ; “two-block $1 / d ^ { \prime }$ (one eigenvalue equal to 1 and $d - 1$ equal to $1 / \kappa )$ Langevin samplers are tuned as in Theorem 1; the diffusion uses the spectrum-agnostic schedule of Proposition 4 $( T = \sqrt { N } , \delta = 1 / N ^ { 2 } )$ . Lines are the quotients of the asymptotic equivalents, ${ \frac { 1 } { 5 } } \kappa _ { \mathrm { e f f } } \ \mathrm { ( U L A ) }$ and ${ \frac { 2 } { 5 } } \kappa _ { \mathrm { e f f } }$ (ULD), with no fitted constant: all four families collapse onto them within $3 . 4 \%$ , consistent with a finite-N correction of order $1 / \log N \approx 3 . 6 \%$ . (b) Budget $N _ { \varepsilon }$ after which the error stays below $\varepsilon = 1 0 ^ { - 3 } \sqrt { \mathrm { T r } \Sigma }$ , where $\sqrt { \mathrm { T r } \Sigma } = W _ { 2 } ( \delta _ { 0 } , \mathbb { P } )$ is the scale of the target, for the log-uniform spectrum. Dashed: the sufficient budgets of Corollary $5 ( r = 1 0 $ in Remark 6 for the diffusion), within a factor 3 of the observed ones. The diffusion budget does not depend on κ. The Langevin budgets follow the $\sqrt { \kappa }$ rate of Table 1 as long as $\varepsilon \lesssim \lambda _ { \mathrm { m i n } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } = \sqrt { \mathrm { T r } \Sigma / \kappa } ;$ ; beyond, the burn-in $N _ { \star } = \Theta ( \kappa \log \kappa )$ of Remark 2 takes over and the budget grows linearly in $\kappa .$ Kinks on the ULA curve mark the switch between the two branches of its step-size.

The ratio $\phi = \eta / \gamma$ of ULD equals the step-size η of ULA. This is expected: for large friction, the position approximately follows the overdamped Langevin dynamics slowed down by a factor γ [Pavliotis, 2014], so that ϕ acts as an effective step-size. The small-friction regime yields the same rate (Corollary B.13): with the EM scheme, neither brings the acceleration expected from ULD.

Remark 2 (Admissibility and burn-in). The conditions on N in Theorem 1 are exactly the requirements under which our analysis operates, namely $\eta \leq \lambda _ { \operatorname* { m i n } }$ for ULA and $\eta \gamma \leq 1$ for ULD (Appendix B). They amount to a burn-in phase: writing

$$
\begin{array} { r } { N _ { \star } ^ { \mathrm { U L A } } : = \operatorname* { m i n } \{ n \geq 3 : n \geq \frac { 1 } { 2 } \kappa \log n \} \quad a n d \quad N _ { \star } ^ { \mathrm { U L D } } : = \operatorname* { m i n } \{ n \geq 3 : n \geq 4 \kappa \log n \} , } \end{array}
$$

provided $N \geq 3 ,$ the condition on N for ULA (resp. ULD) holds ifand only $i f N \ge N _ { \star } ^ { \mathrm { U L A } } ( r e s p . N \ge N _ { \star } ^ { \mathrm { U L D } } )$ , and both thresholds grow as $\Theta ( \kappa \log \kappa )$ . The guarantee thus applies only after $\Theta ( \kappa \log \kappa )$ iterations. Ill-conditioning <sub>penalizes Langevin samplers twice: through the</sub> √<sub>κfactor in the rate, and through the budget required before the</sub> guarantee applies. Neither penalty has a counterpartfor diffusion: Theorem 3 holdsfor every $N \geq 1$

Figure 1 illustrates this burn-in. Below $N _ { \star }$ , the step-size of Theorem 1 violates the admissibility conditions of Remark 2. Since our bounds are convex (with respect to the variable η for ULA, $\phi : = \eta / \gamma$ for ULD), their constrained minimizer is the largest admissible value, $\eta = \lambda _ { \mathrm { m i n } }$ and $\eta \gamma = 1$ respectively (Appendix B). This step is dictated by the stiffest direction of the target, so the flattest ones barely move: the error stays at its initial value $W _ { 2 } ( \mathbb { P } _ { 0 } , \mathbb { P } ) = { \sqrt { \operatorname { T r } \Sigma } }$ for $N \lesssim \kappa ,$ , then decays like $e ^ { - c N / \kappa } \left( c = 2 \right.$ for ULA, $c = 1 / 4$ for ULD) down to the discretization bias of that step, reached at $N \approx N _ { \star } = \Theta ( \kappa \log \kappa )$ . Only then does the log $N / N$ regime of Theorem 1 begin, whereas the diffusion sampler follows its asymptote from $N \approx 1 0 ^ { 2 }$ onwards.

## 2.3 Diffusion models: circumventing the condition-number dependence

In contrast, diffusion models follow the scores $s _ { t } ,$ whose conditioning $\kappa _ { t }$ tends to 1 at large times, and achieve rates that depend on neither κ nor $\lambda _ { \operatorname* { m i n } }$

Theorem 3 (Diffusion sampler). Consider an iteration budget $N \geq 1$ . Set horizon $T = ( \mathrm { T r } ( \Sigma ^ { 5 } ) / \mathrm { T r } \Sigma ) ^ { 1 / 4 } { \sqrt { N } }$ , earlystopping threshold $\delta = \mathrm { T r } \Sigma / ( d N ^ { 2 } )$ , and geometric time discretization grid $\tau _ { k } = T ( \delta / T ) ^ { k / N } f o r k \in \{ 0 , \dots , N \}$ Then the diffusion algorithm (6) yields

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \le \frac { \sqrt { \mathrm { T r } \Sigma } } { N } \Big ( \frac { 5 } { 8 } a _ { N } \rho ^ { 1 / N } \log ( \rho ^ { 2 / 5 } N ) + 1 + \sqrt { 8 } \Big ) , \quad \rho : = \frac { d \lambda _ { \operatorname* { m a x } } } { \mathrm { T r } \Sigma } \in [ 1 , d ] ,
$$

where $a _ { N } : = ( 1 + 4 N ^ { - 2 } ) ^ { 1 / 2 } N ^ { 5 / ( 2 N ) }$ satisfies $a _ { N } \leq 2 ^ { 7 / 4 }$ for all $N \geq 1$ and $a _ { N }  1$ . Moreover, this bound is asymptotically exact: under the same hyperparameters,

$$
{ \cal W } _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \underset { N  \infty } { \sim } \frac { 5 } { 8 } \sqrt { \mathrm { T r } \Sigma } \frac { \log N } { N } ,\tag{7}
$$

so that,forfixed Σ, the ratio ofthe bound to the actual error tends to 1.

The previous theorem gives a spectrum-adaptive bound. The simplified $O ( \sqrt { \mathrm { T r } \Sigma }$ log $N / N )$ rate is obtained when d grows at most polynomially with N. The following proposition shows that this growth assumption can be removed at the price of a coarser dependence on the spectrum.

Proposition 4. With the alternative choice $T = \lambda _ { \operatorname* { m a x } } \sqrt { N } ,$ , early-stopping threshold $\delta = \lambda _ { \operatorname* { m a x } } / N ^ { 2 }$ , and geometric time discretization grid $\tau _ { k } = T ( \delta / T ) ^ { k / N }$ for $k \in \{ 0 , \ldots , N \}$ , the diffusion algorithm (6) satisfies, for every $N \geq 1$

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \frac { \sqrt { d \lambda _ { \operatorname* { m a x } } } } { N } \biggr ( \frac { 5 } { 2 ^ { 5 / 4 } } \log N + \Bigl ( \frac { 5 } { 4 } \Bigr ) ^ { 3 / 2 } + 1 \biggr ) .
$$

On the one hand, Theorem 3 provides our sharpest, spectrum-adaptive guarantee, with a prefactor governed by the effective scale $\sqrt { \mathrm { T r } \Sigma }$ . However, achieving this sharp rate requires choosing the horizon $T$ and early-stopping threshold δ using spectral information about Σ, and its simplified $O ( \sqrt { \mathrm { T r } \Sigma } \frac { \mathrm { l o g } \breve { N } } { N } )$ form is stated under the growth condition $d \leq P ( N )$ for some polynomial P. On the other hand, Proposition 4 holds unconditionally in $d ,$ at the price of a coarser prefactor. Since Tr $\Sigma \leq d \lambda _ { \operatorname* { m a x } }$ (with equality if and only if the target is isotropic), the coarse bound exceeds the spectral bound by a factor ${ \sqrt { \rho } } \in [ 1 , { \sqrt { d } } ]$ . Importantly, this gap is merely a finite-N phenomenon reflecting specific hyperparameter choices rather than a different asymptotic sampling regime. We show in Appendix C.4 that any geometric schedule with $T \propto \sqrt { N }$ and $\delta \propto N ^ { - 2 }$ yields the same asymptotic equivalent (7): the rate is robust to schedule tuning. This covers the parameter choices of both Theorem 3 and Proposition 4.

As in Theorem 1, the proof of Theorem 3 and Proposition 4, given in Appendix C, exploits the Gaussian structure of the target distribution. Since the resulting distribution $\mathbb { P } _ { N }$ is also Gaussian, the analysis reduces to the eigenvalues of its covariance matrix, which can be characterized explicitly. This leads to a Wasserstein error bound with three distinct contributions: a discretization error, an early-stopping error controlled by the threshold δ, and an initialization error arising from starting the process at $\mathbb { P } _ { 0 } = \mathcal { N } ( 0 , T \mathrm { I } )$ rather than at the true marginal $\mathcal { N } ( 0 , \Sigma + T \mathrm { I } )$ , which is not directly tractable. The resulting hyperparameter choices and convergence rates reflect a trade-off between these three sources of error. Consistently with the intuition of Section 2.1, the geometric grid $\tau _ { k } = T ( \delta / T ) ^ { k / N }$ takes coarse steps while $s _ { t }$ remains close to isotropic, and only refines them as $t  \delta .$ , when the anisotropy of the true score re-emerges. Such geometrically decreasing step-sizes near the data are standard in the analysis of diffusion models: the schedule of Benton et al. [2024], for instance, combines constant steps far from the data with steps decaying geometrically as $t  0$ . We adopt the geometric grid over the whole horizon $[ \delta , T ]$ , which makes the computations of Appendix C easier.

The rates above describe the sampling error for a fixed iteration budget N. Sampling guarantees are, however, more commonly stated in the reverse direction in the literature: given a target accuracy, how many iterations are required? We follow this convention below.

Corollary 5 (ε-complexity). Let $0 < \varepsilon \leq \sqrt { \mathrm { T r } \Sigma } .$ . In order for the sampling error to satisfy ${ \cal W } _ { 2 } ( { \mathbb P } _ { N } , { \mathbb P } ) \le \varepsilon ,$ it suffices

• for ULA with η = min ${ \left( \lambda _ { \operatorname* { m i n } } , \varepsilon / \sqrt { \operatorname { T r } ( \Sigma ^ { - 1 } ) } \right) }$ , to take

$$
N \geq \frac { \kappa } { 2 } \operatorname* { m a x } \left( 1 , \frac { \lambda _ { \operatorname* { m i n } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } { \varepsilon } \right) \log \frac { \sqrt { 8 \mathrm { T r } \Sigma } } { \varepsilon } ,
$$

• for ULD with η = min $( \sqrt { \lambda _ { \operatorname* { m i n } } / 8 } , \frac { 7 \sqrt { 2 } } { 6 } \varepsilon / \sqrt { \lambda _ { \operatorname* { m i n } } \operatorname { T r } ( \Sigma ^ { - 1 } ) } )$ and $\gamma = \sqrt { 8 / \lambda _ { \operatorname* { m i n } } }$ , to take

$$
N \geq 4 \kappa \operatorname* { m a x } \left( 1 , { \frac { 3 } { 1 4 } } { \frac { \lambda _ { \operatorname* { m i n } } { \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } } { \varepsilon } } \right) \log { \frac { 1 6 { \sqrt { \mathrm { T r } \Sigma } } } { \varepsilon } } ,
$$

• for the diffusion algorithm with $\delta = \varepsilon ^ { 2 } / ( 9 d ) , T = \sqrt { 3 / \varepsilon } \operatorname { T r } \left( ( \Sigma + \delta \operatorname { I } ) ^ { 3 } \Sigma ^ { 2 } \right) ^ { 1 / 4 }$ , and the geometric grid $\tau _ { k } = T ( \delta / T ) ^ { k / N }$ , to take

$$
N \geq { \frac { 3 e } { 4 } } { \frac { \sqrt { \mathrm { T r } \Sigma + 4 \varepsilon ^ { 2 } / 9 } } { \varepsilon } } \log { \frac { T } { \delta } } .
$$

Using Tr $\Sigma \leq d \lambda _ { \mathrm { m a x } }$ and $\operatorname { T r } ( \Sigma ^ { - 1 } ) \leq d / \lambda _ { \operatorname* { m i n } }$ , we retrieve the ε-complexities of Table $1 , i . e . , \tilde { O } ( \kappa + \sqrt { d \kappa \lambda _ { \mathrm { m a x } } } / \varepsilon )$ for Langevin dynamics versus the condition-number-free $\tilde { O } ( \sqrt { d \lambda _ { \operatorname* { m a x } } } / \varepsilon )$ for diffusion. The spectral form is more informative. For the Langevin samplers, the $1 / \varepsilon$ rate only sets in once $\varepsilon \leq \lambda _ { \operatorname* { m i n } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) }$ for ULA and $\begin{array} { r } { \varepsilon \leq \frac { 3 } { 1 4 } \lambda _ { \operatorname* { m i n } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } \end{array}$ for ULD; above these thresholds, the budget is a burn-in of order $\kappa \log ( \sqrt { \mathrm { T r } \Sigma } / \varepsilon )$ , in line with Remark 2. The diffusion sampler, on the contrary, has no burn-in: its budget is proportional to $1 / \varepsilon$ for every $\varepsilon \leq \sqrt { \mathrm { T r } \Sigma }$ , up to the factor log $( T / \delta ) = O ( \log ( { \sqrt { \mathrm { T r } \Sigma } } / \varepsilon ) )$ , which does not involve $\kappa .$ Finally, the constants are better for ULA than for ULD; this reflects our proof technique rather than an intrinsic advantage, as we did not seek to optimize them.

Remark 6. The restriction $\varepsilon \leq \sqrt { \mathrm { T r } \Sigma }$ is harmless: $\sqrt { \operatorname { T r } \Sigma } = W _ { 2 } ( \delta _ { 0 } , \mathbb { P } )$ is the error of the trivial output 0, so larger error tolerances require no iterations at all. Moreover, the constant 3e/4 ofthe diffusion budget can be tradedfor a mild burn-in. We show in Appendix D that, with the same δ, T and grid, it suffices to take,for any $r \geq 1$

$$
N \geq r \operatorname* { m a x } \left( 1 , \frac { 3 e ^ { 1 / r } } { 4 r } \frac { \sqrt { \mathrm { T r } \Sigma + 4 \varepsilon ^ { 2 } / 9 } } { \varepsilon } \right) \log \frac { T } { \delta } .
$$

For $r = 1$ the maximum is always attained by its second argument, which gives the corollary. As r grows, the constant $3 e ^ { 1 / r } / 4$ decreases towards $3 / 4 ,$ , while thefirst argument, a κ-free burn-in ofr $\log ( T / \delta )$ iterations, prevails $f o r \varepsilon > c _ { r } \sqrt { \mathrm { T r } \Sigma } ,$ , with $c _ { r } = \textstyle { \frac { 3 } { 2 } } { \bigl ( } 4 r ^ { 2 } e ^ { - 2 / r } - 1 { \bigr ) } ^ { - 1 / 2 }$

## 3 Learning: the impact of noise on the implicit bias

The previous results assume the score of the target distribution—or, for diffusion, its noisy versions $s _ { t } { - } \mathbf { \imath } 0$ be known exactly. In practice, these scores must first be learned from (noisy) samples. The goal of this section is to study whether the noise introduced in diffusion models leads to a different score estimator. We are given a dataset $\mathcal { D } = \{ x _ { i } \} _ { 1 \leq i \leq m }$ of m independent realizations of P and denote by $\begin{array} { r } { \hat { \Sigma } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } x _ { i } x _ { i } ^ { \top } } \end{array}$ the empirical covariance matrix. Since the score of a Gaussian distribution is linear, we consider the score model

$$
s _ { A } : x \mapsto - A x ,
$$

where $A \in \mathbb { R } ^ { d \times d }$ , which recovers the true score s when $A = \Sigma ^ { - 1 }$ . When $\hat { \Sigma }$ is invertible, which is almost surely the case if $m \geq d ,$ , the score-matching loss [Hyvarinen¨ , 2005]

$$
L : A \mapsto { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } { \Big ( } \nabla \cdot s _ { A } ( x _ { i } ) + { \frac { 1 } { 2 } } \| s _ { A } ( x _ { i } ) \| ^ { 2 } { \Big ) } ,\tag{8}
$$

where ∥·∥ denotes the Euclidean norm, admits the unique minimizer $\hat { \Sigma } ^ { - 1 }$ . In the overparameterized regime $m < d ,$ however, it has no minimizer as it diverges towards $- \infty$ along Ker $\hat { \Sigma } .$ In this regime, the direction of divergence followed by a given procedure is not determined by L alone but also depends on the optimization algorithm and its initialization, which is usually referred to as an implicit bias of the optimization algorithm [Gunasekar et al., 2018]. Our goal is to compare the implicit biases of two algorithms. First, we study gradient descent initialized at $A _ { 0 } = 0$ and write $A _ { K }$ for the parameter obtained after $K \geq 1$ gradient steps on (8). We then compare with the iterate $A _ { K } ^ { \sigma }$ obtained by running K steps of gradient descent, from the same initialization, on a diffusion-style noised objective. Fix a noise level $\sigma > 0$ and let $\omega \sim \mathcal { N } ( 0 , \sigma ^ { 2 } \mathrm { I } )$ . The noised objective is

$$
L _ { \sigma } : A \mapsto \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathbb { E } _ { \omega } \big [ \nabla \cdot s _ { A } ( x _ { i } + \omega ) + \frac { 1 } { 2 } \| s _ { A } ( x _ { i } + \omega ) \| ^ { 2 } \big ] .
$$

As shown by Vincent [2011], this coincides, up to an A-independent constant, with the standard denoising scorematching regression objective. Additionally, a direct computation shows that

$$
L _ { \sigma } ( A ) = L ( A ) + \frac { \sigma ^ { 2 } } { 2 } \| A \| _ { F } ^ { 2 } .
$$

Thus, it is equivalent to adding a ridge-type regularization term.

We show that, in this Gaussian setting, noise injection does not fundamentally change the learned score: both procedures recover the empirical covariance $\hat { \Sigma }$ on its range, up to a shift of order $\sigma ^ { 2 }$ , and differ only in how they regularize its kernel. The key takeaway is that the benefit of the noising process identified in the sampling phase has no counterpart in the learning phase, at least in the Gaussian setting considered here.

Since A plays the role of the precision matrix $\Sigma ^ { - 1 }$ in the score model $s _ { A } ( x ) = - A x$ , Theorem 7 below characterizes, up to an exponentially decaying error, the covariance matrices estimated by each procedure—namely $A _ { K } ^ { - 1 }$ and $( A _ { K } ^ { \sigma } ) ^ { - 1 }$ . Let us denote by $\hat { \lambda } _ { \mathrm { m a x } }$ the largest eigenvalue of $\hat { \Sigma } , \hat { \lambda } _ { \mathrm { m i n } }$ its smallest eigenvalue, $\hat { \lambda } _ { \operatorname* { m i n } } ^ { + }$ its smallest positive eigenvalue and $\Pi _ { \mathrm { K e r } \hat { \Sigma } }$ the orthogonal projector onto its kernel, of dimension $\operatorname* { m a x } ( 0 , d - m )$ almost surely.

Theorem 7 (Score learning). Consider gradient descent initialized at $A _ { 0 } = 0$ , with step-size $\alpha > 0$ , on both L and $L _ { \sigma }$ . Let $K \geq 1$

$\begin{array} { r } { I f \alpha < \frac { 2 } { \hat { \lambda } _ { \mathrm { m a x } } } } \end{array}$ , the iterate $A _ { K }$ is invertible and

$$
A _ { K } ^ { - 1 } = \hat { \Sigma } + \frac { 1 } { K \alpha } \Pi _ { \mathrm { K e r } \hat { \Sigma } } + O ( \rho _ { 1 } ^ { K } ) ,
$$

with $\rho _ { 1 } = \operatorname* { m a x } \big ( | 1 - \alpha \widehat { \lambda } _ { \operatorname* { m i n } } ^ { + } | , | 1 - \alpha \widehat { \lambda } _ { \operatorname* { m a x } } | \big ) < 1 .$

$\begin{array} { r } { I f \alpha < \frac { 2 } { \hat { \lambda } _ { \mathrm { m a x } } + \sigma ^ { 2 } } } \end{array}$ , the iterate $A _ { K } ^ { \sigma }$ is invertible and

$$
( A _ { K } ^ { \sigma } ) ^ { - 1 } = \hat { \Sigma } + \sigma ^ { 2 } { \mathrm { I } } + O ( \rho _ { 2 } ^ { K } ) ,
$$

$$
w i t h \rho _ { 2 } = \operatorname* { m a x } \big ( | 1 - \alpha ( \widehat { \lambda } _ { \operatorname* { m i n } } + \sigma ^ { 2 } ) | , | 1 - \alpha ( \widehat { \lambda } _ { \operatorname* { m a x } } + \sigma ^ { 2 } ) | \big ) < 1 .
$$

The distinction between the two procedures is clearest when $\hat { \Sigma }$ is singular, that is, almost surely when $m < d .$ For the standard score-matching loss $L$ , the resulting non-identifiability is resolved implicitly by gradient descent: up to an exponentially small optimization error, $A _ { K } ^ { - 1 }$ coincides with $\hat { \Sigma }$ on its range, while a correction of order $1 / ( K \alpha )$ appears only in the kernel of $\hat { \Sigma }$ . In contrast, the noised objective $L _ { \sigma }$ induces an explicit ridge regularization, and this translates into a uniform covariance shift $\sigma ^ { 2 }$ I: all directions are regularized by the same amount. Thus the two procedures recover a regularized empirical covariance matrix, but differ slightly in the regularization: the implicit bias of $L$ acts only in directions that are not identified by the data, whereas noise injection regularizes all directions indiscriminately. Yet, in order to recover the true covariance directions, one should take $\sigma ^ { 2 } \ll \hat { \lambda } _ { \operatorname* { m i n } } ^ { + }$ and, respectively, $1 / ( K \alpha ) \ll \hat { \lambda } _ { \operatorname* { m i n } } ^ { + }$ . In this regime, both regularizations are essentially identical, meaning that the noising process does not help regularize the estimated score compared to learning the original score (with early stopping). We finally note that this constitutes an instance where the gradient descent path follows the explicit ridge regularization path, which is a structure appearing in many contexts [Ali et al., 2020].

## 4 Conclusion

This work demonstrates an advantage of modeling the score at various levels of noise in the Gaussian setting: more efficient sampling by gradually adding anisotropy. This benefit is specific to sampling: in the same setting, the score learned with noise coincides, up to the regularization of the empirical nullspace, with the one learned without. Our sampling results take the form of scaling laws, both for the sampling error and for the hyperparameters that achieve it. We emphasize that, despite its simplicity, the Gaussian framework enables a rigorous proof by characterizing the exact dependence on the condition number. Weaker assumptions, $e . g .$ , log-concavity, lead to loose dependence on the condition number, which is insufficient for our purpose.

Natural directions for future work include incorporating the score estimation error into the sampling analysis and extending the comparison beyond the Gaussian setting.

## Acknowledgements

This work has received support from the French government, managed by the National Research Agency, under the France 2030 program with the reference “PR[AI]RIE-PSAI” (ANR-23-IACL-0008), and through the Gen´ e-Pi project,´ as part of the PEPR Artificial Intelligence programme (ANR-25-PEIA-0006).

## References

Alnur Ali, Edgar Dobriban, and Ryan Tibshirani. The implicit regularization of stochastic gradient flow for least squares. In International Conference on Machine Learning (ICML), pages 233–244, 2020. (cited on page 10)

Brian D. O. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3): 313–326, 1982. (cited on page 4)

Simon Apers, Sander Gribling, and Daniel Szil ´ agyi. Hamiltonian Monte Carlo for efficient Gaussian sampling: long´ and random steps. Journal of Machine Learning Research, 25(348):1–30, 2024. (cited on page 2)

Joe Benton, Valentin De Bortoli, Arnaud Doucet, and George Deligiannidis. Nearly d-linear convergence bounds for diffusion models via stochastic localization. In International Conference on Learning Representations (ICLR), 2024. (cited on pages 4 and 8)

Eliot Beyler and Francis Bach. Convergence of deterministic and stochastic diffusion-model samplers: a simple analysis in Wasserstein distance. arXiv preprint arXiv:2508.03210, 2025. (cited on page 3)

Xiang Cheng, Niladri S. Chatterji, Peter L. Bartlett, and Michael I. Jordan. Underdamped Langevin MCMC: a non-asymptotic analysis. In Conference on Learning Theory (COLT), pages 300–323, 2018. (cited on pages 2 and 3)

Arnak S. Dalalyan. Theoretical guarantees for approximate sampling from smooth and log-concave densities. Journal ofthe Royal Statistical Society Series B: Statistical Methodology, 79(3):651–676, 2017. (cited on page 3)

Arnak S. Dalalyan and Lionel Riou-Durand. On sampling from a log-concave density using kinetic Langevin diffusions. Bernoulli, 26(3):1956–1988, 2020. (cited on pages 2 and 3)

Valentin De Bortoli. Convergence of denoising diffusion models under the manifold hypothesis. Transactions on Machine Learning Research, 2022. (cited on page 4)

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, pages 8780–8794, 2021. (cited on page 1)

Alain Durmus, Szymon Majewski, and Błazej Miasojedow. Analysis of Langevin Monte Carlo via convex optimization.˙ Journal of Machine Learning Research, 20(73):1–46, 2019. (cited on page 2)

David J. Field. Relations between the statistics of natural images and the response properties of cortical cells. Journal ofthe Optical Society ofAmerica A, 4(12):2379–2394, 1987. (cited on page 1)

Matthias Gelbrich. On a formula for the L<sup>2</sup>-Wasserstein metric between measures on Euclidean and Hilbert spaces. Mathematische Nachrichten, 147(1):185–203, 1990. (cited on pages 3 and 16)

Nicola¨ı Gouraud, Pierre Le Bris, Adrien Majka, and Pierre Monmarche. HMC and underdamped Langevin united in´ the unadjusted convex smooth case. SIAM/ASA Journal on Uncertainty Quantification, 13(1):278–303, 2025. (cited on page 2)

Suriya Gunasekar, Jason Lee, Daniel Soudry, and Nathan Srebro. Characterizing implicit bias in terms of optimization geometry. In International Conference on Machine Learning (ICML), 2018. (cited on page 9)

Florentin Guth, Simon Coste, Valentin De Bortoli, and Stephane Mallat. Wavelet score-based generative modeling. In´ Advances in Neural Information Processing Systems (NeurIPS), volume 35, pages 478–491, 2022. (cited on page 1)

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 6840–6851, 2020. (cited on page 1)

Emiel Hoogeboom, Victor Garcia Satorras, Clement Vignac, and Max Welling. Equivariant diffusion for molecule´ generation in 3D. In International Conference on Machine Learning (ICML), pages 8867–8887, 2022. (cited on page 1)

Samuel Hurault, Matthieu Terris, Thomas Moreau, and Gabriel Peyre. From score matching to diffusion: a fine-grained´ error analysis in the Gaussian setting. arXiv preprint arXiv:2503.11615, 2025. (cited on page 2)

Samuel Hurault, Thomas Moreau, and Gabriel Peyre. Geometry-aware discretization error of diffusion models.´ arXiv preprint arXiv:2605.08392, 2026. (cited on page 2)

Aapo Hyvarinen. Estimation of non-normalized statistical models by score matching. ¨ Journal of Machine Learning Research, 6(24):695–709, 2005. (cited on page 9)

Eliahu I. Jury. A simplified stability criterion for linear discrete systems. Proceedings of the IRE, 50(6):1493–1500, 1962. (cited on page 24)

Kyurae Kim, Samuel Gruffaz, Ji Won Park, and Alain Oliviero Durmus. Analysis of kinetic Langevin Monte Carlo under the stochastic exponential Euler discretization from underdamped all the way to overdamped. Electronic Journal of Statistics, 20(2):3106–3142, 2026. (cited on page 2)

Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. DiffWave: A versatile diffusion model for audio synthesis. In International Conference on Learning Representations (ICLR), 2021. (cited on page 1)

Ben Leimkuhler and Charles Matthews. Molecular Dynamics: With Deterministic and Stochastic Numerical Methods, volume 39 of Interdisciplinary Applied Mathematics. Springer, 2015. (cited on page 1)

Benedict Leimkuhler, Daniel Paulin, and Peter A. Whalley. Contraction and convergence rates for discretized kinetic Langevin dynamics. SIAM Journal on Numerical Analysis, 62(3):1226–1258, 2024. (cited on page 2)

Grigorios A. Pavliotis. Stochastic Processes and Applications: Diffusion Processes, the Fokker–Planck and Langevin Equations, volume 60 of Texts in Applied Mathematics. Springer, 2014. (cited on pages 1 and 6)

Fabian Pedregosa. On the convergence of the unadjusted Langevin algorithm. http://fa.bianp.net/blog/ 2023/ulaq/, 2023. (cited on pages 2 and 4)

Emile Pierret and Bruno Galerne. Diffusion models for Gaussian distributions: Exact solutions and Wasserstein errors. In International Conference on Machine Learning (ICML), pages 49355–49381, 2025. (cited on page 2)

Gareth O. Roberts and Richard L. Tweedie. Exponential convergence of Langevin distributions and their discrete approximations. Bernoulli, 2(4):341–363, 1996. (cited on page 1)

Ruoqi Shen and Yin Tat Lee. The randomized midpoint method for log-concave sampling. In Advances in Neural Information Processing Systems (NeurIPS), volume 32, 2019. (cited on page 2)

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning (ICML), pages 2256–2265, 2015. (cited on pages 1 and 4)

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems (NeurIPS), volume 32, 2019. (cited on page 1)

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Scorebased generative modeling through stochastic differential equations. In International Conference on Learning Representations (ICLR), 2021. (cited on pages 1 and 4)

Stanislas Strasman, Antonio Ocello, Claire Boyer, Sylvain Le Corff, and Vincent Lemaire. An analysis of the noise schedule for score-based generative models. Transactions on Machine Learning Research, 2025. (cited on page 3)

Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Computation, 23(7): 1661–1674, 2011. (cited on page 9)

Joseph L. Watson, David Juergens, Nathaniel R. Bennett, Brian L. Trippe, Jason Yim, Helen E. Eisenach, Woody Ahern, Andrew J. Borst, Robert J. Ragotte, Lukas F. Milles, et al. De novo design of protein structure and function with RFdiffusion. Nature, 620(7976):1089–1100, 2023. (cited on page 1)

Andre Wibisono. Sampling as optimization in the space of measures: The Langevin dynamics as a composite optimization problem. In Conference on Learning Theory (COLT), 2018. (cited on page 2)

A Preliminaries 14   
B Proof of Theorem 1 (Langevin samplers) 17   
B.1 Overview 17   
B.2 Unadjusted Langevin algorithm . 18   
B.3 Underdamped Langevin dynamics 22   
C Proof of Theorem 3 and Proposition 4 (diffusion sampler) 32   
C.1 Overview 32   
C.2 Parametric bound for the diffusion sampler 33   
C.3 Choice of hyperparameters 37   
C.4 Asymptotic equivalent 39   
D Proof of Corollary 5 (ε-complexity) 41   
E Proof of Theorem 7 (score learning) 45   
E.1 Losses and gradients 45   
E.2 Closed form of the iterates 46   
E.3 Inverses of the iterates 47   
LLM usage 49   
Code availability 49

## A Preliminaries

We start with a lemma that serves for all three algorithms. Throughout, the constant random vector 0 is regarded as a degenerate centered Gaussian vector with a vanishing covariance matrix; this convention lets us handle the zero initializations of ULA and ULD without special treatment.

Lemma A.1. Let P be an orthogonal matrix diagonalizing Σ, i.e $\boldsymbol { \cdot } , \boldsymbol { \Sigma } = P \boldsymbol { \Lambda } P ^ { \intercal }$ with $\boldsymbol { \Lambda } = \mathrm { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { d } )$ , and let A(Σ) denote the set ofmatrices that are diagonal in the eigenbasis ofΣ:

$$
\begin{array} { r } { A ( \Sigma ) = \{ P D P ^ { \top } , D d i a g o n a l \} . } \end{array}
$$

For $M \in { \mathcal { A } } ( \Sigma )$ , we write $[ M ] _ { i } f o r$ its i-th diagonal entry in this basis. Let $( A _ { k } ^ { x x } , A _ { k } ^ { x v } , A _ { k } ^ { v x } , A _ { k } ^ { v v } , B _ { k } ^ { x } , B _ { k } ^ { v } ) _ { k \geq 0 }$ be elements $o f \mathcal { A } ( \Sigma ) , ( \xi _ { k } )$ a sequence ofindependent standard Gaussian vectors in $\mathbb { R } ^ { d }$ , and $( X _ { 0 } , V _ { 0 } )$ a centered, possibly degenerate, Gaussian pair, independent $o f \left( \xi _ { k } \right)$ , whose covariance blocks $\mathbb { E } [ X _ { 0 } X _ { 0 } ^ { \top } ] , \mathbb { E } [ V _ { 0 } V _ { 0 } ^ { \top } ]$ and $\mathbb { E } [ X _ { 0 } V _ { 0 } ^ { \top } ]$ all belong to $\boldsymbol { \mathcal { A } } ( \Sigma )$ . Consider the process $( X _ { k } , V _ { k } )$ defined by

$$
\forall k \geq 0 , \left\{ \begin{array} { l l } { X _ { k + 1 } = A _ { k } ^ { x x } X _ { k } + A _ { k } ^ { x v } V _ { k } + B _ { k } ^ { x } \xi _ { k } , } \\ { V _ { k + 1 } = A _ { k } ^ { v x } X _ { k } + A _ { k } ^ { v v } V _ { k } + B _ { k } ^ { v } \xi _ { k } , } \end{array} \right.\tag{9}
$$

and set $\Sigma _ { k } : = \mathbb { E } [ X _ { k } X _ { k } ^ { \top } ] , S _ { k } : = \mathbb { E } [ V _ { k } V _ { k } ^ { \top } ]$ and $\Gamma _ { k } : = \mathbb { E } [ X _ { k } V _ { k } ^ { \top } ]$ . Finally, for $1 \leq i \leq d ,$ denote by $x _ { k } ^ { ( i ) } , v _ { k } ^ { ( i ) }$ and $\zeta _ { k } ^ { ( i ) }$ the i-th coordinates of $P ^ { \top } X _ { k } , P ^ { \top } V _ { k }$ and $P ^ { \top } \xi _ { k }$ respectively. Then:

(i) $( X _ { k } , V _ { k } )$ is a centered Gaussian vector, and $\Sigma _ { k } , S _ { k }$ and $\Gamma _ { k }$ belong to ${ \mathcal { A } } ( \Sigma )$

(ii) The d pairs $( x _ { k } ^ { ( i ) } , v _ { k } ^ { ( i ) } ) , 1 \leq i \leq d ,$ are mutually independent, and each ofthemfollows the scalar recursion

$$
\forall k \geq 0 , \left\{ \begin{array} { l l } { x _ { k + 1 } ^ { ( i ) } = [ A _ { k } ^ { x x } ] _ { i } x _ { k } ^ { ( i ) } + [ A _ { k } ^ { x v } ] _ { i } v _ { k } ^ { ( i ) } + [ B _ { k } ^ { x } ] _ { i } \zeta _ { k } ^ { ( i ) } , } \\ { v _ { k + 1 } ^ { ( i ) } = [ A _ { k } ^ { v x } ] _ { i } x _ { k } ^ { ( i ) } + [ A _ { k } ^ { v v } ] _ { i } v _ { k } ^ { ( i ) } + [ B _ { k } ^ { v } ] _ { i } \zeta _ { k } ^ { ( i ) } , } \end{array} \right.
$$

where the $( \zeta _ { k } ^ { ( i ) } ) _ { k , i }$ are mutually independent standard one-dimensional Gaussian variables, independent of $( X _ { 0 } , V _ { 0 } )$

(iii) The eigenvalues $\lambda _ { i } ^ { ( k ) } o f \Sigma _ { k }$ are the variances ofthe variables $\boldsymbol { x } _ { k } ^ { ( i ) }$ :

$$
\lambda _ { i } ^ { ( k ) } = \mathrm { V a r } ( x _ { k } ^ { ( i ) } ) .
$$

Remark A.2. ULA (2), ULD (4) and the diffusion sampler (6) are all instances of the process $( X _ { k } , V _ { k } )$ above. For ULD the identification is immediate: $A _ { k } ^ { x x } = \mathrm { I } , A _ { k } ^ { x v } = \eta \mathrm { I } , B _ { k } ^ { x } = 0 , A _ { k } ^ { v x } = - \eta \Sigma ^ { - 1 } , A _ { k } ^ { v v } = ( 1 - \eta \gamma ) ^ { \top }$ I and $B _ { k } ^ { v } = \sqrt { 2 \eta \gamma } ]$ . ULA and the diffusion have no velocity component, but they fit the same framework upon taking $V _ { 0 } = 0$ and $A _ { k } ^ { x v } = A _ { k } ^ { v x } = A _ { k } ^ { v v } = B _ { k } ^ { v } = 0$ . ULA then corresponds to $A _ { k } ^ { x x } = \mathrm { I } - \eta \Sigma ^ { - 1 }$ and $B _ { k } ^ { x } = \sqrt { 2 \eta } \mathrm { I }$ , and the diffusion to $A _ { k } ^ { x x } = \mathrm { I } - \eta _ { k + 1 } C _ { \tau _ { k } } ^ { - 1 }$ and $B _ { k } ^ { x } = \sqrt { \eta _ { k + 1 } } { \mathrm { I } }$ , the covariance T I ofits initialization $\mathcal { N } ( 0 , T \mathrm { I } )$ being indeed in A(Σ).

ProofofLemma A.1. We first prove (ii). For every $k \geq 0 ,$ , set $\tilde { X } _ { k } : = P ^ { \top } X _ { k } , \tilde { V } _ { k } : = P ^ { \top } V _ { k }$ and $\tilde { \xi } _ { k } : = P ^ { \top } \xi _ { k }$ . Since P is orthogonal, $( \tilde { \xi } _ { k } ) _ { k \geq 0 }$ is again a sequence of independent standard Gaussian vectors, independent of $( X _ { 0 } , V _ { 0 } )$ ; hence the family $( \zeta _ { k } ^ { ( i ) } ) _ { k \geq 0 , 1 \leq i \leq d }$ consists of mutually independent $\mathcal { N } ( 0 , 1 )$ variables, independent of $( X _ { 0 } , V _ { 0 } )$ , which is the last claim of (ii). Multiplying (9) by $P ^ { \top }$ and inserting $P P ^ { \top } = \operatorname { I }$ in front of $X _ { k } , V _ { k }$ and $\xi _ { k } \ \mathrm { g i v e s }$

$$
\begin{array} { r } { \left\{ \tilde { X } _ { k + 1 } = P ^ { \top } A _ { k } ^ { x x } P \tilde { X } _ { k } + P ^ { \top } A _ { k } ^ { x v } P \tilde { V } _ { k } + P ^ { \top } B _ { k } ^ { x } P \tilde { \xi } _ { k } , \right. } \\ { \left. \tilde { V } _ { k + 1 } = P ^ { \top } A _ { k } ^ { v x } P \tilde { X } _ { k } + P ^ { \top } A _ { k } ^ { v v } P \tilde { V } _ { k } + P ^ { \top } B _ { k } ^ { v } P \tilde { \xi } _ { k } . \right. } \end{array}
$$

By definition of $\boldsymbol { \mathcal { A } } ( \Sigma )$ , every matrix in this system is diagonal, so reading it coordinate by coordinate yields exactly the announced recursion

$$
\begin{array} { r l r } & { \{ \boldsymbol { x } _ { k + 1 } ^ { ( i ) } = [ A _ { k } ^ { x x } ] _ { i } \boldsymbol { x } _ { k } ^ { ( i ) } + [ A _ { k } ^ { x v } ] _ { i } \boldsymbol { v } _ { k } ^ { ( i ) } + [ B _ { k } ^ { x } ] _ { i } \zeta _ { k } ^ { ( i ) } , \quad \quad } & { 1 \le i \le d . } \\ & { \boldsymbol { v } _ { k + 1 } ^ { ( i ) } = [ A _ { k } ^ { v x } ] _ { i } \boldsymbol { x } _ { k } ^ { ( i ) } + [ A _ { k } ^ { v v } ] _ { i } \boldsymbol { v } _ { k } ^ { ( i ) } + [ B _ { k } ^ { v } ] _ { i } \zeta _ { k } ^ { ( i ) } , \quad \quad } & { } \end{array}
$$

It remains to show that the d pairs $( x _ { k } ^ { ( i ) } , v _ { k } ^ { ( i ) } )$ are independent, which we do by induction on k.

Base case. $( \tilde { X } _ { 0 } , \tilde { V } _ { 0 } )$ is a centered Gaussian vector, as $( X _ { 0 } , V _ { 0 } )$ is, and its covariance matrix

$$
\begin{array} { r l } { \left[ P ^ { \top } \Sigma _ { 0 } P } & { { } P ^ { \top } \Gamma _ { 0 } P \right] } \\ { \left[ P ^ { \top } \Gamma _ { 0 } ^ { \top } P } & { { } P ^ { \top } S _ { 0 } P \right] } \end{array}
$$

has four diagonal blocks, since $\Sigma _ { 0 } , S _ { 0 }$ and $\Gamma _ { 0 }$ belong to $\boldsymbol { \mathcal { A } } ( \Sigma )$ . The pairs $( x _ { 0 } ^ { ( i ) } , v _ { 0 } ^ { ( i ) } ) _ { 1 \leq i \leq d }$ are therefore uncorrelated and, being jointly Gaussian, independent.

Induction step. Assume the pairs $( x _ { k } ^ { ( i ) } , v _ { k } ^ { ( i ) } ) _ { 1 \leq i \leq d }$ are independent for some $k \geq 0$ . The vector $\xi _ { k }$ is independent of $X _ { 0 } , V _ { 0 }$ and $( \xi _ { j } ) _ { j < k }$ , of which $( X _ { k } , V _ { k } )$ is a deterministic function; hence $\xi _ { k }$ is independent of $( X _ { k } , V _ { k } )$ , and $\tilde { \xi } _ { k }$ of $( \tilde { X } _ { k } , \tilde { V } _ { k } )$ . As the coordinates $\zeta _ { k } ^ { ( 1 ) } , \ldots , \zeta _ { k } ^ { ( d ) }$ of $\tilde { \xi } _ { k }$ are themselves independent, the d triplets $( x _ { k } ^ { ( i ) } , v _ { k } ^ { ( i ) } , \zeta _ { k } ^ { ( i ) } )$ are mutually independent. Each pair $( x _ { k + 1 } ^ { ( i ) } , v _ { k + 1 } ^ { ( i ) } )$ is a deterministic function of the i-th triplet, so the pairs at step $k + 1$ are independent as well. This completes the induction, and the proof of (ii).

We now turn to (i). The vector $( X _ { k + 1 } , V _ { k + 1 } )$ is a linear function of $\left( X _ { k } , V _ { k } , \xi _ { k } \right)$ . If $( X _ { k } , V _ { k } )$ is a centered Gaussian vector, so is $( X _ { k } , V _ { k } , \xi _ { k } )$ , because $\xi _ { k }$ is centered Gaussian and independent of $( X _ { k } , V _ { k } )$ (see the proof of (ii)); hence $( X _ { k + 1 } , V _ { k + 1 } )$ is centered Gaussian too, and the claim follows by induction from the assumption on $( X _ { 0 } , V _ { 0 } )$ . As for the covariance matrices, $\Sigma _ { k }$ belongs to ${ \mathcal { A } } ( \Sigma )$ if and only if $P ^ { \top } \Sigma _ { k } P$ is diagonal. But $P ^ { \top } \Sigma _ { k } P$ is the covariance matrix of $\tilde { X } _ { k }$ , whose $( i , j )$ entry is $\mathrm { C o v } ( x _ { k } ^ { ( i ) } , x _ { k } ^ { ( j ) } )$ , which vanishes for $i \neq j \mathrm { \ b y \ ( i i ) }$ . Thus $\Sigma _ { k } \in \mathcal { A } ( \Sigma )$ and the same argument applies to $S _ { k }$ and $\Gamma _ { k }$ .

Finally, (iii) is a direct consequence of the above: the eigenvalues of $\Sigma _ { k }$ are the diagonal entries of $P ^ { \top } \Sigma _ { k } P .$ namely the variances $\mathrm { V a r } ( x _ { k } ^ { ( i ) } )$ . □

We also rely on the following elementary fact, which expresses the Wasserstein distance between two centered Gaussians whose covariances are diagonal in the same orthonormal basis in terms of their eigenvalues.

Lemma A.3. Let $\Sigma _ { 1 } , \Sigma _ { 2 } \in \mathcal { A } ( \Sigma )$ be positive semi-definite, with diagonal entries (hence eigenvalues) $( a _ { i } ) _ { 1 \leq i \leq d }$ and $( b _ { i } ) _ { 1 \leq i \leq d }$ in the basis P. Then

$$
W _ { 2 } \big ( \mathcal { N } ( 0 , \Sigma _ { 1 } ) , \mathcal { N } ( 0 , \Sigma _ { 2 } ) \big ) = \left( \sum _ { i = 1 } ^ { d } \Big ( \sqrt { a _ { i } } - \sqrt { b _ { i } } \Big ) ^ { 2 } \right) ^ { 1 / 2 } .
$$

Proof. Since $\Sigma _ { 1 }$ and $\Sigma _ { 2 }$ belong to ${ \mathcal { A } } ( \Sigma )$ , they commute, and the formula recalled in Section 2.1 applies [Gelbrich, 1990]:

$$
W _ { 2 } ( { \mathcal N } ( 0 , \Sigma _ { 1 } ) , { \mathcal N } ( 0 , \Sigma _ { 2 } ) ) = \| { \Sigma _ { 1 } ^ { 1 / 2 } - \Sigma _ { 2 } ^ { 1 / 2 } } \| _ { F } .
$$

Now $\Sigma _ { 1 } ^ { 1 / 2 } = P \mathrm { d i a g } ( \sqrt { a _ { 1 } } , \ldots , \sqrt { a _ { d } } ) P ^ { \top }$ and $\Sigma _ { 2 } ^ { 1 / 2 } = P \mathrm { d i a g } ( \sqrt { b _ { 1 } } , \ldots , \sqrt { b _ { d } } ) P ^ { \top }$ , and the Frobenius norm is invariant under orthogonal conjugation, whence the result. □

Finally, the following elementary lemma is what we use to optimize the bounds of Propositions B.1 and B.7.

Lemma A.4. Let A, B, C and D be positive constants, and let

$$
\begin{array} { r } { \begin{array} { r c l } { f \colon ( 0 , D ] } & { \longrightarrow } & { { \mathbb R } , } \\ { x } & { \longmapsto } & { A e ^ { - B x } + C x . } \end{array} } \end{array}
$$

Then f is strictly convex and, provided $0 <$ log $\begin{array} { r } { \frac { A B } { C } \leq B D } \end{array}$ , it admits a unique minimizer,

$$
x _ { * } = \frac { 1 } { B } \log \frac { A B } { C } , \quad w i t h \quad f ( x _ { * } ) = \frac { C } { B } \bigg ( 1 + \log \frac { A B } { C } \bigg ) .
$$

Proof. Strict convexity follows from $f ^ { \prime \prime } > 0 \mathrm { o n } \left( 0 , D \right]$ . Solving $f ^ { \prime } ( x ) = 0 \mathrm { g i v e s } x _ { \ast }$ , the condition $\begin{array} { r } { 0 < \log { \frac { A B } { C } } \leq B D } \end{array}$ is exactly what ensures $0 < x _ { * } \le D$ , and the value of $f ( x _ { * } )$ follows by substitution. □

## B Proof of Theorem 1 (Langevin samplers)

## B.1 Overview

This section is devoted to the proof of the following theorem, stated in Section 2.2.

Theorem 1 (Langevin samplers). Let $N \geq 2$ be the iteration budget, and set the step-size η (and the friction $\gamma f o r$ ULD) as

$\begin{array} { r } { U L A \colon \eta = \frac { 1 } { 2 } \lambda _ { \operatorname* { m a x } } \frac { \log N } { N } } \end{array}$ , provided $N \geq { \textstyle { \frac { 1 } { 2 } } } \kappa \log N $

$\begin{array} { r } { U L D ; \eta = \sqrt { 2 \lambda _ { \operatorname* { m a x } } \kappa } \frac { \log N } { N } } \end{array}$ and $\gamma = \sqrt { 8 / \lambda _ { \operatorname* { m i n } } } ,$ provided N ≥ 4κ log N.

Then

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \frac { \lambda _ { \operatorname* { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } { N } \left( C _ { 1 } \log N + C _ { 2 } \right) ,
$$

with $\begin{array} { r } { C _ { 1 } = \frac { 1 } { 4 } , C _ { 2 } = \sqrt { 2 } } \end{array}$ for ULA and $\begin{array} { r } { C _ { 1 } = \frac { 3 } { 7 } , C _ { 2 } = 8 f o r U L D } \end{array}$ . Furthermore, for fixed Σ, these bounds are tight up to a constantfactor:

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \underset { N  \infty } { \sim } C _ { 3 } \lambda _ { \operatorname* { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \frac { \log N } { N } ,
$$

with $\begin{array} { r } { C _ { 3 } = \frac { 1 } { 8 } f o r U L A } \end{array}$ and $\begin{array} { r } { C _ { 3 } = \frac { 1 } { 4 } f o r U L D . } \end{array}$

Error decomposition. ULA and ULD define time-homogeneous Markov chains. Under stability conditions on the hyperparameters, established in step (i) below, these chains converge, not to the target P, but to an invariant law $\mathbb { P } _ { \infty }$ that depends on the step-size η (and on the friction $\gamma$ for ULD). The triangle inequality

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \le W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } _ { \infty } ) + W _ { 2 } ( \mathbb { P } _ { \infty } , \mathbb { P } )\tag{10}
$$

separates two contributions. The first term is the early-stopping error: after N iterations, the chain has not yet reached its invariant law. The second is the discretization error: the bias of $\mathbb { P } _ { \infty }$ with respect to $\mathbb { P } ,$ , introduced by the Euler–Maruyama scheme.

Reduction to eigenvalues. Lemma A.1 applies to both algorithms (Remark A.2): each $\mathbb { P } _ { N }$ is a centered Gaussian whose covariance $\Sigma _ { N }$ lies in $\boldsymbol { \mathcal { A } } ( \Sigma )$ , and so does the covariance $\Sigma _ { \infty }$ of $\mathbb { P } _ { \infty }$ , the limit of $\left( \Sigma _ { N } \right)$ . By Lemma A.3, both terms in (10) then depend only on the eigenvalues $\lambda _ { i } ^ { ( N ) }$ of $\Sigma _ { N } , \lambda _ { i } ^ { \infty }$ of $\Sigma _ { \infty }$ and $\lambda _ { i }$ of Σ:

$$
W _ { 2 } ^ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } _ { \infty } ) = \sum _ { i = 1 } ^ { d } \Big ( \sqrt { \lambda _ { i } ^ { ( N ) } } - \sqrt { \lambda _ { i } ^ { \infty } } \Big ) ^ { 2 } , \quad W _ { 2 } ^ { 2 } ( \mathbb { P } _ { \infty } , \mathbb { P } ) = \sum _ { i = 1 } ^ { d } \Big ( \sqrt { \lambda _ { i } ^ { \infty } } - \sqrt { \lambda _ { i } } \Big ) ^ { 2 } .\tag{11}
$$

This is the payoff of the Gaussian setting: the analysis reduces to real sequences for which closed forms are available.

Throughout, a choice of hyperparameters is called admissible if it lies in the stability range of the chain, shrunk by numerical constants that give the estimates some slack, keeping the denominators involved bounded away from zero. Remarks B.4 and B.11 make the gap with the exact stability conditions explicit.

Outline. Sections B.2 (ULA) and B.3 (ULD) follow the same four steps.

(i) Closed form. Compute the eigenvalues $\lambda _ { i } ^ { ( N ) }$ and $\lambda _ { i } ^ { \infty }$ from the scalar recursions of Lemma A.1, and identify the stability conditions under which $\left( \Sigma _ { N } \right)$ converges.

(ii) Parametric bound. Bound each of the two terms in (10) and sum: this gives a bound on $W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } )$ valid over the whole admissible range identified in (i) (Proposition B.1 for ULA, Proposition B.7 for ULD).

(iii) Choice of hyperparameters. Minimize, up to a numerical constant, the bound obtained in (ii), and check that the selected values satisfy the stability conditions as soon as $N \geq N _ { \star }$

(iv) Asymptotic equivalent. For the hyperparameters of step (iii), derive an exact equivalent of $W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } )$ as $N \to \infty$ from the closed forms of step (i). This equivalent has the same order log $N / N$ and the same spectral prefactor as the bound of step (iii), up to a numerical constant: the bound is of the right order of magnitude.

## B.2 Unadjusted Langevin algorithm

The goal of this section is to prove the following proposition, which bounds $W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } )$ for every small enough step-size η. To lighten notation, we drop the index i whenever we argue at a fixed eigendirection i.

Proposition B.1 (Parametric bound for $\mathrm { U L A } )$ . For any step-size $0 < \eta \leq \lambda _ { \mathrm { m i n } }$ and any $N \geq 0$ , the error ofULA satisfies

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \sqrt { 2 \operatorname { T r } \Sigma } e ^ { - 2 \eta N / \lambda _ { \operatorname* { m a x } } } + \frac { 1 } { 2 } \sqrt { \operatorname { T r } ( \Sigma ^ { - 1 } ) } \eta .
$$

The two terms of this bound come from bounding the early-stopping error and the discretization error respectively. The first decays exponentially in $N \eta$ , the time at which the process is stopped, while the second grows linearly with the step-size η. The step-size selected in Theorem 1 is the best compromise between these two competing effects.

## B.2.1 Closed form of the eigenvalues

We first establish a closed form for the eigenvalues $\lambda _ { i } ^ { ( N ) }$ and their limit $\lambda _ { i } ^ { \infty }$

Lemma B.2. Assume $0 < \eta < 2 \lambda _ { \mathrm { m i n } }$ and set

$$
\lambda _ { i } ^ { \infty } : = \frac { \lambda _ { i } } { 1 - \frac { \eta } { 2 } \lambda _ { i } ^ { - 1 } } .
$$

Then, for every $N \geq 0 _ { i }$

$$
\lambda _ { i } ^ { ( N ) } = \lambda _ { i } ^ { \infty } \left( 1 - \left( 1 - \frac { \eta } { \lambda _ { i } } \right) ^ { 2 N } \right) ,
$$

and in particular $\lambda _ { i } ^ { ( N ) } \to \lambda _ { i } ^ { \infty }$ as $N \to \infty$

Proof. Fix a direction i and consider the corresponding sequence $( \lambda ^ { ( k ) } )$ of eigenvalues of $\Sigma _ { k }$ (the index i is dropped, as announced). By Lemma A.1(iii) and the identification of Remark $\mathrm { A } . 2 , \lambda ^ { ( k ) }$ is the variance of $x _ { k }$ , where $x _ { 0 } = 0$ and

$$
\forall k \geq 0 , \quad x _ { k + 1 } = \left( 1 - { \frac { \eta } { \lambda } } \right) x _ { k } + { \sqrt { 2 \eta } } \zeta _ { k } ,
$$

with $\left( \zeta _ { k } \right)$ independent standard one-dimensional Gaussians. Since $x _ { k }$ is a function of $( \zeta _ { j } ) _ { j < k }$ , it is independent of $\zeta _ { k }$ and taking variances gives $\lambda ^ { ( 0 ) } = 0$ and

$$
\forall k \geq 0 , \quad \lambda ^ { ( k + 1 ) } = \left( 1 - \frac { \eta } { \lambda } \right) ^ { 2 } \lambda ^ { ( k ) } + 2 \eta ,
$$

a first-order affine recursion. Its ratio satisfies

$$
\left( 1 - \frac { \eta } { \lambda } \right) ^ { 2 } < 1 ,\tag{12}
$$

because $0 < \eta < 2 \lambda _ { \mathrm { m i n } }$ gives $0 < \eta / \lambda < 2 , i . e . , - 1 < 1 - \eta / \lambda < 1$ . In particular the ratio is not 1, so the recursion has a unique fixed point

$$
\ell = \frac { 2 \eta } { 1 - ( 1 - \eta \lambda ^ { - 1 } ) ^ { 2 } } = \frac { 2 \eta } { 2 \eta \lambda ^ { - 1 } - \eta ^ { 2 } \lambda ^ { - 2 } } = \frac { \lambda } { 1 - \frac { \eta } { 2 } \lambda ^ { - 1 } } = \lambda ^ { \infty } ,
$$

and

$$
\forall k \geq 0 , \quad \lambda ^ { ( k ) } = \left( 1 - \frac { \eta } { \lambda } \right) ^ { 2 k } ( \lambda ^ { ( 0 ) } - \ell ) + \ell = \lambda ^ { \infty } \left( 1 - \left( 1 - \frac { \eta } { \lambda } \right) ^ { 2 k } \right) .
$$

Taking $k = N$ gives the closed form, and (12) gives the convergence $\lambda ^ { ( N ) } \to \lambda ^ { \infty }$ . Observe that $\lambda ^ { \infty } \to \lambda$ as $\eta  0 \colon$ the finer the Euler–Maruyama discretization, the smaller the bias. □

## B.2.2 Early-stopping error

We now bound the early-stopping error of ULA.

Lemma B.3. Assume $0 < \eta \leq \lambda _ { \operatorname* { m i n } }$ . Then, for every $N \geq 0$

$$
\begin{array} { r } { W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } _ { \infty } ) \le \sqrt { 2 \operatorname { T r } \Sigma } e ^ { - 2 \eta N / \lambda _ { \operatorname* { m a x } } } . } \end{array}
$$

Proof. Starting from the first identity in (11) and writing ${ \sqrt { a } } - { \sqrt { b } } = ( a - b ) / ( { \sqrt { a } } + { \sqrt { b } } )$

$$
W _ { 2 } ^ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } _ { \infty } ) = \sum _ { i = 1 } ^ { d } \left( \frac { \lambda _ { i } ^ { \infty } - \lambda _ { i } ^ { ( N ) } } { \sqrt { \lambda _ { i } ^ { \infty } } + \sqrt { \lambda _ { i } ^ { ( N ) } } } \right) ^ { 2 } \leq \sum _ { i = 1 } ^ { d } \frac { \big ( \lambda _ { i } ^ { \infty } - \lambda _ { i } ^ { ( N ) } \big ) ^ { 2 } } { \lambda _ { i } ^ { \infty } } .\tag{13}
$$

Since $\lambda ^ { ( N ) } \to \lambda ^ { \infty }$ , this step only costs a factor 2 asymptotically (on $W _ { 2 }$ itself, once the square root is taken). By Lemma B.2, $\begin{array} { r } { \lambda ^ { \infty } - \lambda ^ { ( N ) } = \lambda ^ { \infty } \left( 1 - \frac { \eta } { \lambda } \right) ^ { 2 N } } \end{array}$ , hence

$$
\frac { ( \lambda ^ { \infty } - \lambda ^ { ( N ) } ) ^ { 2 } } { \lambda ^ { \infty } } = \lambda ^ { \infty } \left( 1 - \frac { \eta } { \lambda } \right) ^ { 4 N } .
$$

The assumption $\eta \leq \lambda _ { \mathrm { m i n } }$ gives both $\begin{array} { r } { 0 \leq 1 - \frac { \eta } { \lambda } \leq 1 - \frac { \eta } { \lambda _ { \operatorname* { m a x } } } } \end{array}$ and $\begin{array} { r } { 1 - \frac { \eta } { 2 } \lambda ^ { - 1 } \geq \frac { 1 } { 2 } , i . e . , 0 < \lambda ^ { \infty } \leq 2 \lambda } \end{array}$ , so that

$$
\frac { ( \lambda ^ { \infty } - \lambda ^ { ( N ) } ) ^ { 2 } } { \lambda ^ { \infty } } \leq 2 \lambda \left( 1 - \frac { \eta } { \lambda _ { \operatorname* { m a x } } } \right) ^ { 4 N } .
$$

Summing over i and plugging into (13) yields

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } _ { \infty } ) \leq \sqrt { 2 \operatorname { T r } \Sigma } \left( 1 - \frac { \eta } { \lambda _ { \operatorname* { m a x } } } \right) ^ { 2 N } \leq \sqrt { 2 \operatorname { T r } \Sigma } e ^ { - 2 \eta N / \lambda _ { \operatorname* { m a x } } } ,
$$

where the last inequality uses $1 - u \leq e ^ { - u }$ together with the positivity of $1 - \eta / \lambda _ { \mathrm { m a x } }$

Remark B.4. Lemma B.2 only requires $\eta < 2 \lambda _ { \mathrm { m i n } }$ , which is what the convergence of $\lambda ^ { ( N ) }$ to $\lambda ^ { \infty }$ needs, whereas Lemma B.3 assumes $\eta \leq \lambda _ { \mathrm { m i n } }$ . The latter is a convenience assumption: it keeps $1 - \eta \lambda ^ { - 1 }$ nonnegative and, more importantly, keeps the denominator $1 - \frac { \eta } { 2 } \lambda ^ { - 1 }$ bounded away from zero. Since η is meant to go to 0 with N to reduce the discretization error, this is a mild restriction.

## B.2.3 Discretization error

We turn to the discretization error.

Lemma B.5. Assume $0 < \eta \leq \lambda _ { \mathrm { m i n } }$ . Then

$$
W _ { 2 } ( \mathbb { P } _ { \infty } , \mathbb { P } ) \leq \frac { 1 } { 2 } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \eta .
$$

Proof. This time we start from the second identity in (11):

$$
W _ { 2 } ^ { 2 } ( \mathbb { P } _ { \infty } , \mathbb { P } ) = \sum _ { i = 1 } ^ { d } \left( \frac { \lambda _ { i } ^ { \infty } - \lambda _ { i } } { \sqrt { \lambda _ { i } ^ { \infty } } + \sqrt { \lambda _ { i } } } \right) ^ { 2 } \leq \frac { 1 } { 4 } \sum _ { i = 1 } ^ { d } \frac { ( \lambda _ { i } ^ { \infty } - \lambda _ { i } ) ^ { 2 } } { \lambda _ { i } } ,\tag{14}
$$

where the inequality uses

$$
\lambda ^ { \infty } > \lambda ,\tag{15}
$$

a consequence of $\begin{array} { r } { 1 - \frac { \eta } { 2 } \lambda ^ { - 1 } < 1 } \end{array}$ . By Lemma B.2,

$$
\lambda ^ { \infty } - \lambda = \lambda \left( \frac { 1 } { 1 - \frac { \eta } { 2 } \lambda ^ { - 1 } } - 1 \right) = \frac { \eta } { 2 - \eta \lambda ^ { - 1 } } \leq \eta ,
$$

where we used $\eta \leq \lambda _ { \mathrm { m i n } }$ once more. Since $\lambda ^ { \infty } - \lambda > 0$ by (15), this bound can be squared, and (14) gives the result. □

Proof of Proposition B.1. Combine Lemmas B.3 and B.5 through the triangle inequality (10).

## B.2.4 Choice of the step-size

We now prove the first part of Theorem 1, namely the non-asymptotic bound for ULA.

Proof of Theorem 1, non-asymptotic bound for ULA. Let $\eta _ { N } : = \textstyle { \frac { 1 } { 2 } } \lambda _ { \operatorname* { m a x } } \frac { \log N } { N }$ be the ULA step-size of Theorem 1, and let $N \geq { \textstyle { \frac { 1 } { 2 } } } \kappa \log N$ . This condition is exactly $\eta _ { N } \ \leq \ \lambda _ { \mathrm { m i n } }$ , and since $N > 1$ we have $0 < \eta _ { N } \leq \lambda _ { \operatorname* { m i n } } .$ Proposition B.1 therefore applies with $\eta = \eta _ { N }$ , and since $e ^ { - 2 \eta _ { N } N / \lambda _ { \mathrm { m a x } } } = 1 / N$

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \frac { \sqrt { 2 \operatorname { T r } \Sigma } } { N } + \frac { 1 } { 4 } \lambda _ { \operatorname* { m a x } } \sqrt { \operatorname { T r } ( \Sigma ^ { - 1 } ) } \frac { \log N } { N } .
$$

Finally,

$$
\begin{array} { r } { \sqrt { \mathrm { T r } \Sigma } \leq \lambda _ { \mathrm { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } , } \end{array}\tag{16}
$$

since the left-hand side is at most $\sqrt { d \lambda _ { \operatorname* { m a x } } }$ while the right-hand side is at least $\lambda _ { \operatorname* { m a x } } \sqrt { d / \lambda _ { \operatorname* { m a x } } } = \sqrt { d \lambda _ { \operatorname* { m a x } } }$ . This gives the bound of Theorem 1,

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \frac { \lambda _ { \operatorname* { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } { N } \left( C _ { 1 } \log N + C _ { 2 } \right) ,
$$

with $\begin{array} { r } { C _ { 1 } = \frac { 1 } { 4 } } \end{array}$ and $C _ { 2 } = { \sqrt { 2 } } .$

Remark B.6 (On the choice of η). The step-size η<sub>N</sub> is not the exact minimizer of the bound of Proposition B.1, but an asymptotically equivalent simplification ofit. Indeed, applying Lemma A.4 with

$$
A = \sqrt { 2 \mathrm { T r } \Sigma } , \quad B = \frac { 2 N } { \lambda _ { \mathrm { m a x } } } , \quad C = \frac { 1 } { 2 } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } , \quad D = \lambda _ { \mathrm { m i n } } ,
$$

the infimum ofthat bound is attained at

$$
\eta _ { * } = \frac { \lambda _ { \operatorname* { m a x } } } { 2 N } \log \frac { 2 N A } { \lambda _ { \operatorname* { m a x } } C } = \frac { \lambda _ { \operatorname* { m a x } } } { 2 N } \Bigg ( \log N + \log \left( \frac { 4 \sqrt { 2 } } { \lambda _ { \operatorname* { m a x } } } \sqrt { \frac { \mathrm { T r } \Sigma } { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } \right) \Bigg ) ,
$$

which is admissible as soon as $2 N A > \lambda _ { \operatorname* { m a x } } C$ and $\begin{array} { r } { 2 N \geq \kappa \log { \frac { 2 N A } { \lambda _ { \operatorname* { m a x } } C } } } \end{array}$ , and the minimum equals

$$
f ( \eta _ { * } ) = \frac { \lambda _ { \operatorname* { m a x } } C } { 2 N } \Bigl ( 1 + \log \frac { 2 N A } { \lambda _ { \operatorname* { m a x } } C } \Bigr ) .
$$

The choice $\eta _ { N }$ amounts to keeping only the log N term in the parentheses. As a result, $f ( \eta _ { N } )$ and $f ( \eta _ { * } )$ share the leading term $\frac { \lambda _ { \operatorname* { m a x } } C } { 2 } \frac { \log N } { N }$ and differ by $O ( 1 / N )$ ; in particular $f ( \eta _ { N } ) / f ( \eta _ { * } ) \to 1$ . The only loss is on the constant $C _ { 2 } ,$ in exchange for a step-size that requires no spectral information beyond $\lambda _ { \operatorname* { m a x } } ,$ and an admissibility condition that depends on κ only.

## B.2.5 Asymptotic equivalent

We finally prove the second part of Theorem 1 for ULA: with the step-size $\begin{array} { r } { \eta = \eta _ { N } = \frac { 1 } { 2 } \lambda _ { \mathrm { m a x } } \frac { \log N } { N } } \end{array}$

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \underset { N  \infty } { \sim } \frac { 1 } { 8 } \lambda _ { \operatorname* { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \frac { \log N } { N } .
$$

Proof of Theorem 1, asymptotic equivalent for ULA. We take N large enough for $\eta _ { N }$ to be admissible in Proposition B.1. The starting point is the closed form of Lemma B.2,

$$
\lambda ^ { ( N ) } = \frac { \lambda } { 1 - \frac { \eta _ { N } } { 2 } \lambda ^ { - 1 } } - \frac { \lambda } { 1 - \frac { \eta _ { N } } { 2 } \lambda ^ { - 1 } } \left( 1 - \frac { \eta _ { N } } { \lambda } \right) ^ { 2 N } .\tag{17}
$$

We first show that $\lambda ^ { ( N ) } \to \lambda$ . On the one hand, since $\eta _ { N }  0 ,$

$$
\frac { \lambda } { 1 - \frac { \eta _ { N } } { 2 } \lambda ^ { - 1 } } \underset { N  \infty } { \longrightarrow } \lambda .\tag{18}
$$

On the other hand, using $\log ( 1 - u ) \leq - u$ for $u < 1$ and then $\lambda \le \lambda _ { \mathrm { m a x } }$

$$
\left( 1 - \frac { \eta _ { N } } { \lambda } \right) ^ { 2 N } = \exp { \left( 2 N \log { \left( 1 - \frac { \eta _ { N } } { \lambda } \right) } \right) } \leq \exp { \left( - \frac { 2 N \eta _ { N } } { \lambda } \right) } = N ^ { - \lambda _ { \operatorname* { m a x } } / \lambda } \leq \frac { 1 } { N } ,\tag{19}
$$

so the second term in (17) tends to 0, and with (18) we get $\lambda ^ { ( N ) } \to \lambda$ . Applying Lemma A.3 directly to $W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } )$ then gives

$$
W _ { 2 } ^ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) = \sum _ { i = 1 } ^ { d } \Big ( \sqrt { \lambda _ { i } ^ { ( N ) } } - \sqrt { \lambda _ { i } } \Big ) ^ { 2 } = \sum _ { i = 1 } ^ { d } \left( \frac { \lambda _ { i } ^ { ( N ) } - \lambda _ { i } } { \sqrt { \lambda _ { i } ^ { ( N ) } } + \sqrt { \lambda _ { i } } } \right) ^ { 2 } \sim \frac { 1 } { 4 } \sum _ { i = 1 } ^ { d } \frac { ( \lambda _ { i } ^ { ( N ) } - \lambda _ { i } ) ^ { 2 } } { \lambda _ { i } } .\tag{20}
$$

It remains to estimate $\lambda ^ { ( N ) } - \lambda$ . By (17) and (19),

$$
\begin{array} { c } { \lambda ^ { ( N ) } - \lambda = \displaystyle \frac { \lambda } { 1 - \frac { \eta _ { N } } { 2 } \lambda ^ { - 1 } } - \lambda + O \left( \frac { 1 } { N } \right) = \frac { \eta _ { N } } { 2 - \eta _ { N } \lambda ^ { - 1 } } + O \left( \frac { 1 } { N } \right) } \\ { = \displaystyle \frac { 1 } { 2 - \eta _ { N } \lambda ^ { - 1 } } \times \frac { 1 } { 2 } \lambda _ { \operatorname* { m a x } } \frac { \log N } { N } + O \left( \frac { 1 } { N } \right) \sim \frac { 1 } { 4 } \lambda _ { \operatorname* { m a x } } \frac { \log N } { N } , } \end{array}
$$

an equivalent which does not depend on λ. Plugging it into (20),

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \underset { N  \infty } { \sim } \frac { 1 } { 8 } \lambda _ { \operatorname* { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \frac { \log N } { N } ,
$$

which is the announced equivalent, consistent with the non-asymptotic bound.

## B.3 Underdamped Langevin dynamics

The proof of Theorem 1 for ULD follows the structure of the previous section. We first prove the proposition below, which bounds $W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } )$ for all $( \eta , \gamma )$ satisfying one of two sets of assumptions: a large-friction regime, which is the one adopted in Theorem 1, and a small-friction regime. Both lead to similar bounds, up to numerical constants; Remark B.8 explains why the first one is retained in Theorem 1.

Proposition B.7 (Parametric bound for ULD). Let $N \geq 0 .$ . In the large-friction regime

$$
\gamma \geq \sqrt { 8 / \lambda _ { \operatorname* { m i n } } } a n d \eta \leq \frac { 1 } { \gamma } ,\tag{LF}
$$

the error of ULD satisfies

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \le 8 \sqrt { \mathrm { T r } \Sigma } e ^ { - 2 \eta N / ( \lambda _ { \operatorname* { m a x } } \gamma ) } + \frac { 6 } { 7 } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \frac { \eta } { \gamma } ,
$$

and in the small-friction regime

$$
\gamma \leq \sqrt { 2 / \lambda _ { \operatorname* { m a x } } } a n d \eta \leq \frac { 1 } { 2 } \lambda _ { \operatorname* { m i n } } \gamma ,\tag{SF}
$$

it satisfies

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq 8 \sqrt { \mathrm { T r } \Sigma } e ^ { - \eta \gamma N / 2 } + \frac { 3 } { 2 } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \frac { \eta } { \gamma } .
$$

Here too, each bound is the sum of an early-stopping term and a discretization term. Up to a numerical constant, the discretization term has the same expression in both regimes. Only the early-stopping term changes, with the friction $\gamma$ appearing in the denominator or in the numerator of the exponent according to whether it is large or small.

Remark B.8. The small-friction bound ofProposition B.7 is strictly decreasing in γ. There is no trade-off: one simply saturates the constraint $\gamma \leq \sqrt { 2 / \lambda _ { \operatorname* { m a x } } } ,$ , which also makes the admissible range $\eta \leq \frac { 1 } { 2 } \lambda _ { \operatorname* { m i n } } \gamma$ as wide as possible. The resulting rates are the same in both regimes. We retain the large-friction regime (LF) in Theorem 1 because the roots $r _ { \pm }$ introduced below are then real and positive, which keeps the analysis simplest, and because the bound has exactly theform ofthe ULA bound with $\phi : = \eta / \gamma$ in place ofη; with the hyperparameters ofthe theorem, $\begin{array} { r } { \phi = \frac { 1 } { 2 } \lambda _ { \mathrm { m a x } } \frac { \log N } { N } } \end{array}$ is precisely the ULA step-size.

## B.3.1 Closed form of the eigenvalues

As before, we start with a closed form for the eigenvalues of $\Sigma _ { N }$ and $\Sigma _ { \infty }$ .

Lemma B.9. Fix an iteration budget $N \geq 0 .$ . We consider two sets ofassumptions:

• either

$$
\gamma > \frac { 2 } { \sqrt { \lambda _ { \operatorname* { m i n } } } } \quad a n d \quad 0 < \eta < \frac { 2 } { \gamma } ,\tag{LF<sub>0</sub>}
$$

in which case we set

$$
r _ { \pm } = 1 - \frac { 1 } { 2 } \eta \bigl ( \gamma \mp \sqrt { \gamma ^ { 2 } - 4 \lambda ^ { - 1 } } \bigr ) ;\tag{21}
$$

• or

$$
0 < \gamma < \frac { 2 } { \sqrt { \lambda _ { \operatorname* { m a x } } } } \quad a n d \quad 0 < \eta < \lambda _ { \operatorname* { m i n } } \gamma ,\tag{SF<sub>0</sub>}
$$

in which case we set

$$
r _ { \pm } = 1 - \frac { 1 } { 2 } \eta \bigl ( \gamma \mp i \sqrt { 4 \lambda ^ { - 1 } - \gamma ^ { 2 } } \bigr ) ,
$$

where i denotes the imaginary unit.

In both cases, the eigenvalues $\lambda ^ { ( N ) }$ are exactly

$$
\lambda ^ { ( N ) } = \frac { 2 \gamma ( 2 - \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ) } { ( \gamma - \eta \lambda ^ { - 1 } ) ( 4 - 2 \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ) } \lambda - \frac { 2 \eta \gamma } { \gamma ^ { 2 } - 4 \lambda ^ { - 1 } } \left( \frac { r _ { + } ^ { 2 N } } { 1 - r _ { + } ^ { 2 } } - \frac { 2 ( r _ { + } r _ { - } ) ^ { N } } { 1 - r _ { + } r _ { - } } + \frac { r _ { - } ^ { 2 N } } { 1 - r _ { - } ^ { 2 } } \right) ,\tag{22}
$$

and they converge as $N \to \infty$ to

$$
\lambda ^ { \infty } = \frac { 2 \gamma ( 2 - \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ) } { ( \gamma - \eta \lambda ^ { - 1 } ) ( 4 - 2 \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ) } \lambda .\tag{23}
$$

Proof. Fix an index i and consider the corresponding sequence of eigenvalues $( \lambda ^ { ( k ) } )$ . By Lemma A.1(iii) and Remark A.2,

$$
\forall k \geq 0 , \quad \lambda ^ { ( k ) } = \operatorname { V a r } ( x _ { k } ) ,\tag{24}
$$

where $( x _ { k } , v _ { k } )$ is defined by $x _ { 0 } = v _ { 0 } = 0$ and

$$
\left\{ \begin{array} { l l } { x _ { k + 1 } = x _ { k } + \eta v _ { k } , } \\ { v _ { k + 1 } = ( 1 - \eta \gamma ) v _ { k } - \eta \lambda ^ { - 1 } x _ { k } + \sqrt { 2 \eta \gamma } \zeta _ { k } , } \end{array} \right.
$$

with $\left( \zeta _ { k } \right)$ independent standard one-dimensional Gaussians. In matrix form, $z _ { k + 1 } = M z _ { k } + u \zeta _ { k }$ with

$$
z _ { k } = \binom { x _ { k } } { v _ { k } } , \quad M = \left[ \begin{array} { c c } { 1 } & { \eta } \\ { - \eta \lambda ^ { - 1 } } & { 1 - \eta \gamma } \end{array} \right] , \quad u = \left[ \begin{array} { c c } { 0 } \\ { \sqrt { 2 \eta \gamma } } \end{array} \right] ,\tag{25}
$$

so that, since $z _ { 0 } = 0$

$$
z _ { k } = \sum _ { j = 0 } ^ { k - 1 } M ^ { j } u \zeta _ { k - 1 - j } .\tag{26}
$$

Let $\beta _ { j }$ denote the $( 1 , 2 )$ entry of $M ^ { j }$ . Reading the first coordinate of (26) at $k = N$ gives

$$
x _ { N } = \sqrt { 2 \eta \gamma } \sum _ { j = 0 } ^ { N - 1 } \beta _ { j } \zeta _ { N - 1 - j } ,
$$

hence by (24)

$$
\lambda ^ { ( N ) } = 2 \eta \gamma \sum _ { j = 0 } ^ { N - 1 } \beta _ { j } ^ { 2 } .\tag{27}
$$

It remains to compute the $\beta _ { j }$ . By the Cayley–Hamilton theorem, $M ^ { 2 } - \mathrm { T r } ( M ) M + \operatorname* { d e t } ( M ) \mathrm { I } = 0$ , hence $M ^ { j + 2 } = \mathrm { T r } ( M ) M ^ { j + 1 } - \mathrm { d e t } \bar { ( M ) } M ^ { j }$ for all $j ,$ and taking (1, 2) entries,

$$
\forall j \geq 0 , \quad \beta _ { j + 2 } = \operatorname { T r } ( M ) \beta _ { j + 1 } - \operatorname* { d e t } ( M ) \beta _ { j } .\tag{28}
$$

From (25), $\operatorname { T r } ( M ) = 2 - \eta \gamma$ and $\operatorname* { d e t } ( M ) = 1 - \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 }$ , so the behavior of $( \beta _ { j } )$ is governed by the roots of

$$
P ( X ) = X ^ { 2 } - ( 2 - \eta \gamma ) X + 1 - \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ,
$$

whose discriminant is

$$
\Delta = \eta ^ { 2 } ( \gamma ^ { 2 } - 4 \lambda ^ { - 1 } ) .\tag{29}
$$

This formula shows that the two assumptions on $\gamma$ in Lemma B.9 are exactly the cases where $\Delta$ is positive for all eigenvalues (large friction, real roots), or negative for all eigenvalues (small friction, complex roots): these are the two homogeneous regimes we study.

Stability. By (27), for $\lambda ^ { ( N ) }$ to converge the sequence $( \beta _ { j } )$ must tend to 0, i.e., both roots of P must have modulus strictly less than 1. By the Jury stability criterion [Jury, 1962], this amounts to

$$
| \mathrm { d e t } ( M ) | < 1 , \quad P ( 1 ) > 0 , \quad P ( - 1 ) > 0 ,
$$

that is,

$$
| 1 - \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } | < 1 , \quad \eta ^ { 2 } \lambda ^ { - 1 } > 0 , \quad 4 - 2 \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } > 0 .
$$

The second condition holds since $\eta > 0$ . For the third, as $\eta ^ { 2 } \lambda ^ { - 1 } > 0$ , it suffices that $\eta \gamma \leq 2 \colon$ this is immediate under $\mathrm { ( L F _ { 0 } ) }$ , whose assumption on η is precisely the strict version of this inequality, and under $( \mathrm { S F _ { 0 } } )$ it follows from

$$
P ( - 1 ) = \frac { ( \eta - \gamma \lambda ) ^ { 2 } } { \lambda } + 4 - \gamma ^ { 2 } \lambda \geq 4 - \gamma ^ { 2 } \lambda _ { \operatorname* { m a x } } > 0 .
$$

As for the first condition, det $( M ) < 1$ is equivalent to $\eta / \gamma < \lambda$ , which holds since $\eta / \gamma < \lambda _ { \mathrm { m i n } } \leq \lambda$ under both $\left( \mathrm { L F _ { 0 } } \right)$ (where $\eta / \gamma < 2 / \gamma ^ { 2 } < \lambda _ { \mathrm { m i n } } / 2 )$ and $( \mathrm { S F _ { 0 } } ) { \mathrm { ; } }$ ; and det $( M ) > - 1$ follows from $P ( 1 ) + P ( - 1 ) = 2 ( 1 + \operatorname* { d e t } ( M ) )$ ， which is positive by the two conditions just established. Thus, in both cases,

$$
| \mathrm { d e t } ( M ) | < 1 , \quad P ( 1 ) > 0 , \quad P ( - 1 ) > 0 ,\tag{30}
$$

and the assumptions of Lemma B.9 guarantee stability.

Computation $o f \lambda ^ { ( N ) }$ . By (29), P has two distinct roots, which are the $r _ { \pm }$ of the statement: $r _ { \pm } = 1 - { \textstyle \frac { 1 } { 2 } } \eta \big ( \gamma \mp$ $\sqrt { \gamma ^ { 2 } - 4 \lambda ^ { - 1 } } )$ under $\mathrm { ( L F _ { 0 } ) }$ and $r _ { \pm } = 1 - { \textstyle \frac { 1 } { 2 } } \eta \bigl ( \gamma \mp i \sqrt { 4 \lambda ^ { - 1 } - \gamma ^ { 2 } } \bigr )$ under $( \mathrm { { S F _ { 0 } } ) }$ . They satisfy

$$
r _ { + } + r _ { - } = \mathrm { T r } ( M ) = 2 - \eta \gamma\tag{31}
$$

and

$$
r _ { + } r _ { - } = \operatorname* { d e t } ( M ) = 1 - \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ,\tag{32}
$$

two relations we will use repeatedly. By (28), $\beta _ { j } = A r _ { + } ^ { j } + B r _ { - } ^ { j }$ for some constants A, B determined by $\beta _ { 0 } = 0$ and $\beta _ { 1 } = \eta$ , which gives

$$
\forall j \ge 0 , \quad \beta _ { j } = \frac { \eta } { r _ { + } - r _ { - } } ( r _ { + } ^ { j } - r _ { - } ^ { j } ) ,
$$

and with (27),

$$
\lambda ^ { ( N ) } = \frac { 2 \eta ^ { 3 } \gamma } { ( r _ { + } - r _ { - } ) ^ { 2 } } \sum _ { j = 0 } ^ { N - 1 } ( r _ { + } ^ { 2 j } - 2 ( r _ { + } r _ { - } ) ^ { j } + r _ { - } ^ { 2 j } ) .
$$

The Jury criterion, satisfied under (LF ) and $( \mathrm { S F _ { 0 } } )$ , ensures

$$
| r _ { \pm } | < 1 ,\tag{33}
$$

so $r _ { + } ^ { 2 } , r _ { - } ^ { 2 }$ and $r _ { + } r _ { - }$ all differ from 1, and summing the three geometric series yields

$$
\lambda ^ { ( N ) } = \lambda ^ { \infty } - \frac { 2 \eta ^ { 3 } \gamma } { ( r _ { + } - r _ { - } ) ^ { 2 } } \left( \frac { r _ { + } ^ { 2 N } } { 1 - r _ { + } ^ { 2 } } - \frac { 2 ( r _ { + } r _ { - } ) ^ { N } } { 1 - r _ { + } r _ { - } } + \frac { r _ { - } ^ { 2 N } } { 1 - r _ { - } ^ { 2 } } \right) ,
$$

with

$$
\lambda ^ { \infty } = \frac { 2 \eta ^ { 3 } \gamma } { ( r _ { + } - r _ { - } ) ^ { 2 } } \left( \frac { 1 } { 1 - r _ { + } ^ { 2 } } - \frac { 2 } { 1 - r _ { + } r _ { - } } + \frac { 1 } { 1 - r _ { - } ^ { 2 } } \right) .\tag{34}
$$

By (33), $\lambda ^ { ( N ) }$ converges to $\lambda ^ { \infty }$ . It remains to show that $\lambda ^ { \infty }$ has the announced form. Putting the three fractions in (34) over the common denominator $( 1 - r _ { + } ^ { 2 } ) ( 1 - r _ { - } ^ { 2 } ) ( 1 - r _ { + } r _ { - } )$ , the numerator C factorizes:

$$
\begin{array} { l } { C = ( 1 - r _ { + } ^ { 2 } ) ( 1 - r _ { + } r _ { - } ) - 2 ( 1 - r _ { + } ^ { 2 } ) ( 1 - r _ { - } ^ { 2 } ) + ( 1 - r _ { + } r _ { - } ) ( 1 - r _ { - } ^ { 2 } ) } \\ { \quad = r _ { + } ^ { 2 } - 2 r _ { + } r _ { - } + r _ { - } ^ { 2 } + r _ { + } r _ { - } ( r _ { + } ^ { 2 } - 2 r _ { + } r _ { - } + r _ { - } ^ { 2 } ) } \\ { \quad = ( r _ { + } - r _ { - } ) ^ { 2 } ( 1 + r _ { + } r _ { - } ) , } \end{array}
$$

so that

$$
\lambda ^ { \infty } = 2 \eta ^ { 3 } \gamma \frac { 1 + r _ { + } r _ { - } } { ( 1 - r _ { + } ^ { 2 } ) ( 1 - r _ { - } ^ { 2 } ) ( 1 - r _ { + } r _ { - } ) } .\tag{35}
$$

To conclude independently of the regime, we express everything through the sum and product of the roots. On the one hand, by (31) and (32),

$$
\begin{array} { r l } & { ( 1 - r _ { + } ^ { 2 } ) ( 1 - r _ { - } ^ { 2 } ) = 1 + ( r _ { + } r _ { - } ) ^ { 2 } - ( r _ { + } + r _ { - } ) ^ { 2 } + 2 r _ { + } r _ { - } } \\ & { \qquad = 1 + ( 1 - \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ) ^ { 2 } - ( 2 - \eta \gamma ) ^ { 2 } + 2 ( 1 - \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ) } \\ & { \qquad = \eta ^ { 4 } \lambda ^ { - 2 } + 4 \eta ^ { 2 } \lambda ^ { - 1 } - 2 \eta ^ { 3 } \gamma \lambda ^ { - 1 } = \eta ^ { 2 } \lambda ^ { - 1 } ( 4 - 2 \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ) , } \end{array}
$$

the third line being a direct expansion of the second. On the other hand, (32) gives $1 + r _ { + } r _ { - } = 2 - \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 }$ and $1 - r _ { + } r _ { - } = \eta ( \gamma - \eta \lambda ^ { - 1 } )$ . Plugging these into (35),

$$
\lambda ^ { \infty } = \frac { 2 \gamma ( 2 - \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ) } { ( \gamma - \eta \lambda ^ { - 1 } ) ( 4 - 2 \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ) } \lambda ,
$$

which is (23). As expected, this expression does not depend on the regime, and once again $\lambda ^ { \infty } \to \lambda$ as $\eta  0$ . Finally, (22) follows from $( \bar { r } _ { + } - r _ { - } ) ^ { 2 } = ( \bar { r } _ { + } + r _ { - } ) ^ { 2 } - 4 r _ { + } r _ { - } = \eta ^ { 2 } ( \gamma ^ { 2 } - 4 \lambda ^ { - 1 } )$ , by (31) and (32) once more. □

## B.3.2 Early-stopping error

We now bound the early-stopping error of ULD.

Lemma B.10. Let $N \geq 0 .$ . In the large-friction regime (LF), the early-stopping error satisfies

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } _ { \infty } ) \le 8 \sqrt { \mathrm { T r } \Sigma } e ^ { - 2 \eta N / ( \lambda _ { \operatorname* { m a x } } \gamma ) } ,
$$

and in the small-friction regime (SF),

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } _ { \infty } ) \le 8 \sqrt { \mathrm { T r } \Sigma } e ^ { - \eta \gamma N / 2 } .
$$

Proof. Once again, (11) gives

$$
W _ { 2 } ^ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } _ { \infty } ) = \sum _ { i = 1 } ^ { d } \left( \frac { \lambda _ { i } ^ { \infty } - \lambda _ { i } ^ { ( N ) } } { \sqrt { \lambda _ { i } ^ { \infty } } + \sqrt { \lambda _ { i } ^ { ( N ) } } } \right) ^ { 2 } \leq \sum _ { i = 1 } ^ { d } \frac { \big ( \lambda _ { i } ^ { \infty } - \lambda _ { i } ^ { ( N ) } \big ) ^ { 2 } } { \lambda _ { i } ^ { \infty } } .\tag{36}
$$

We first check that

$$
\lambda ^ { \infty } > \lambda .\tag{37}
$$

Starting from (23),

$$
\begin{array} { l } { { \lambda ^ { \infty } - \lambda = \frac { \gamma ( 4 - 2 \eta \gamma + 2 \eta ^ { 2 } \lambda ^ { - 1 } ) - ( \gamma - \eta \lambda ^ { - 1 } ) ( 4 - 2 \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ) } { ( \gamma - \eta \lambda ^ { - 1 } ) ( 4 - 2 \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ) } \lambda } } \\ { { = \frac { \gamma \eta ^ { 2 } \lambda ^ { - 1 } + \eta \lambda ^ { - 1 } ( 4 - 2 \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ) } { ( \gamma - \eta \lambda ^ { - 1 } ) ( 4 - 2 \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ) } \lambda } } \\ { { = \frac { ( 4 - \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ) \eta } { ( \gamma - \eta \lambda ^ { - 1 } ) ( 4 - 2 \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } ) } . } } \end{array}\tag{38}
$$

Recognizing $P ( - 1 ) = 4 - 2 \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 }$ , which we showed to be positive, this reads

$$
\lambda ^ { \infty } - \lambda = \frac { \eta } { \gamma - \eta \lambda ^ { - 1 } } \times \frac { P ( - 1 ) + \eta \gamma } { P ( - 1 ) } .\tag{39}
$$

The difference therefore has the sign of $\gamma - \eta \lambda ^ { - 1 }$ , which is positive by the first Jury condition in (30). This proves (37), and (36) becomes

$$
W _ { 2 } ^ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } _ { \infty } ) \leq \sum _ { i = 1 } ^ { d } \frac { \big ( \lambda _ { i } ^ { \infty } - \lambda _ { i } ^ { ( N ) } \big ) ^ { 2 } } { \lambda _ { i } } .\tag{40}
$$

We are thus led to bound $| \lambda _ { i } ^ { \infty } - \lambda _ { i } ^ { ( N ) } |$ , for which Lemma B.9 provides

$$
| \lambda ^ { \infty } - \lambda ^ { ( N ) } | = \frac { 2 \eta \gamma } { | \gamma ^ { 2 } - 4 \lambda ^ { - 1 } | } \left| \frac { r _ { + } ^ { 2 N } } { 1 - r _ { + } ^ { 2 } } - \frac { 2 ( r _ { + } r _ { - } ) ^ { N } } { 1 - r _ { + } r _ { - } } + \frac { r _ { - } ^ { 2 N } } { 1 - r _ { - } ^ { 2 } } \right| .\tag{41}
$$

Large-friction regime (LF). By (27), $\lambda ^ { \infty }$ is the limit of 2ηγ $\textstyle \sum _ { j = 0 } ^ { N - 1 } \beta _ { j } ^ { 2 }$ as $N \to \infty$ , so the difference $\lambda ^ { \infty } - \lambda ^ { ( N ) }$ is the tail

$$
\lambda ^ { \infty } - \lambda ^ { ( N ) } = 2 \eta \gamma \sum _ { j = N } ^ { \infty } \beta _ { j } ^ { 2 } \geq 0 .
$$

Since moreover $\gamma ^ { 2 } - 4 \lambda ^ { - 1 } > 0$ under (LF), (41) holds without absolute values:

$$
| \lambda ^ { \infty } - \lambda ^ { ( N ) } | = \frac { 2 \eta \gamma } { \gamma ^ { 2 } - 4 \lambda ^ { - 1 } } \left( \frac { r _ { + } ^ { 2 N } } { 1 - r _ { + } ^ { 2 } } - \frac { 2 ( r _ { + } r _ { - } ) ^ { N } } { 1 - r _ { + } r _ { - } } + \frac { r _ { - } ^ { 2 N } } { 1 - r _ { - } ^ { 2 } } \right) .\tag{42}
$$

Now $1 - r _ { + } r _ { - } > 0$ by (33), and (LF) makes the product of the roots positive:

$$
r _ { + } r _ { - } = 1 - \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } > 1 - \eta \gamma \geq 0 .\tag{43}
$$

The middle term in (42) is therefore negative, and

$$
\big | \lambda ^ { \infty } - \lambda ^ { ( N ) } \big | < \frac { 2 \eta \gamma } { \gamma ^ { 2 } - 4 \lambda ^ { - 1 } } \left( \frac { r _ { + } ^ { 2 N } } { 1 - r _ { + } ^ { 2 } } + \frac { r _ { - } ^ { 2 N } } { 1 - r _ { - } ^ { 2 } } \right) ,\tag{44}
$$

where every numerator and denominator is positive: (LF) for $\gamma ^ { 2 } - 4 \lambda ^ { - 1 }$ , and (33) for $1 - r _ { \pm } ^ { 2 }$ . We next compare $| r _ { + } |$ and $| r _ { - } |$ . By (43) the two roots have the same sign, namely that of their sum $2 - \eta \gamma > 0$ by (31) and (LF), so both are positive, and their expression (21) gives

$$
r _ { + } = | r _ { + } | > | r _ { - } | = r _ { - } .\tag{45}
$$

Hence (44) yields

$$
\left| \lambda ^ { \infty } - \lambda ^ { ( N ) } \right| < \frac { 2 \eta \gamma } { \gamma ^ { 2 } - 4 \lambda ^ { - 1 } } \left( \frac { 1 } { 1 - r _ { + } ^ { 2 } } + \frac { 1 } { 1 - r _ { - } ^ { 2 } } \right) r _ { + } ^ { 2 N } .
$$

One could bound the two terms in the parenthesis separately and get a slightly sharper estimate, but this would only improve a numerical constant, so we simply use $1 - \bar { r } _ { - } ^ { 2 } > \mathrm { { i } } - r _ { + } ^ { 2 }$ , a consequence of (45), to obtain

$$
| \lambda ^ { \infty } - \lambda ^ { ( N ) } | < \frac { 4 \eta \gamma } { \gamma ^ { 2 } - 4 \lambda ^ { - 1 } } \times \frac { r _ { + } ^ { 2 N } } { 1 - r _ { + } ^ { 2 } } .
$$

By (LF), $\gamma ^ { 2 } - 4 \lambda ^ { - 1 } \ge \gamma ^ { 2 } - 4 \lambda _ { \operatorname* { m i n } } ^ { - 1 } \ge \frac { 1 } { 2 } \gamma ^ { 2 }$ , hence

$$
| \lambda ^ { \infty } - \lambda ^ { ( N ) } | < \frac { 8 \eta } { \gamma } \times \frac { r _ { + } ^ { 2 N } } { 1 - r _ { + } ^ { 2 } } .\tag{46}
$$

Again without optimizing constants, since $r _ { + } > 0$ 2

$$
\frac { 1 } { 1 - r _ { + } ^ { 2 } } = \frac { 1 } { ( 1 + r _ { + } ) ( 1 - r _ { + } ) } \leq \frac { 1 } { 1 - r _ { + } } ,\tag{47}
$$

and by (21),

$$
\frac { 1 } { 1 - r _ { + } } = \frac { 2 } { \eta \bigl ( \gamma - \sqrt { \gamma ^ { 2 } - 4 \lambda ^ { - 1 } } \bigr ) } = \frac { 2 \bigl ( \gamma + \sqrt { \gamma ^ { 2 } - 4 \lambda ^ { - 1 } } \bigr ) } { \eta \bigl ( \gamma ^ { 2 } - ( \gamma ^ { 2 } - 4 \lambda ^ { - 1 } ) \bigr ) } < \frac { \gamma \lambda } { \eta } .
$$

Combining with (46) and (47),

$$
| \lambda ^ { \infty } - \lambda ^ { ( N ) } | < 8 \lambda r _ { + } ^ { 2 N } .\tag{48}
$$

To make this bound usable we finally bound $r _ { + }$ . The concavity inequality ${ \sqrt { 1 - u } } \leq 1 - { \textstyle { \frac { 1 } { 2 } } } u$ , valid for $u \in [ 0 , 1 ]$ gives

$$
r _ { + } = 1 - \frac { 1 } { 2 } \eta \gamma \left( 1 - \sqrt { 1 - 4 \lambda ^ { - 1 } \gamma ^ { - 2 } } \right) \leq 1 - \frac { 1 } { 2 } \eta \gamma \times \frac { 2 } { \lambda \gamma ^ { 2 } } = 1 - \frac { \eta } { \gamma \lambda } \leq 1 - \frac { \eta } { \gamma \lambda _ { \operatorname* { m a x } } } .
$$

Since $1 - \eta / ( \gamma \lambda _ { \mathrm { m a x } } )$ is positive under (LF), raising to the power 2N and using, as in Section B.2, the convexity inequality

$$
\forall u \in \mathbb { R } , \quad 1 - u \leq e ^ { - u } ,\tag{49}
$$

we deduce from (48)

$$
\vert \lambda ^ { \infty } - \lambda ^ { ( N ) } \vert < 8 \lambda \left( 1 - \frac { \eta } { \gamma \lambda _ { \operatorname* { m a x } } } \right) ^ { 2 N } \le 8 \lambda e ^ { - 2 \eta N / ( \gamma \lambda _ { \operatorname* { m a x } } ) } .\tag{50}
$$

Small-friction regime (SF). We return to (41). The roots $r _ { \pm }$ are now complex conjugates, so

$$
| r _ { + } | ^ { 2 } = | r _ { - } | ^ { 2 } = r _ { + } r _ { - } = 1 - \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } = : R ^ { 2 } ,
$$

the common value being the constant coefficient of $P .$ Moreover $\frac { r _ { + } ^ { 2 N } } { 1 - r _ { + } ^ { 2 } }$ and $\frac { r _ { - } ^ { 2 N } } { 1 - r _ { - } ^ { 2 } }$ are conjugate as well, hence

$$
\begin{array} { l } { \displaystyle \left| \frac { r _ { + } ^ { 2 N } } { 1 - r _ { + } ^ { 2 } } - \frac { 2 ( r _ { + } r _ { - } ) ^ { N } } { 1 - r _ { + } r _ { - } } + \frac { r _ { - } ^ { 2 N } } { 1 - r _ { - } ^ { 2 } } \right| = 2 \left| \operatorname { R e } \left( \frac { r _ { + } ^ { 2 N } } { 1 - r _ { + } ^ { 2 } } \right) - \frac { R ^ { 2 N } } { 1 - R ^ { 2 } } \right| } \\ { \leq 2 \left| \frac { r _ { + } ^ { 2 N } } { 1 - r _ { + } ^ { 2 } } \right| + \frac { 2 R ^ { 2 N } } { 1 - R ^ { 2 } } = 2 R ^ { 2 N } \left( \frac { 1 } { | 1 - r _ { + } ^ { 2 } | } + \frac { 1 } { 1 - R ^ { 2 } } \right) } \\ { \leq \frac { 4 R ^ { 2 N } } { 1 - R ^ { 2 } } , } \end{array}
$$

where the last step uses $| 1 - r _ { + } ^ { 2 } | \geq 1 - | r _ { + } | ^ { 2 } = 1 - R ^ { 2 }$ . Now $R ^ { 2 } = 1 - \eta ( \gamma - \eta \lambda ^ { - 1 } )$ and, by (SF),

$$
\gamma - \eta \lambda ^ { - 1 } \geq \gamma - \frac { 1 } { 2 } \lambda _ { \operatorname* { m i n } } \gamma \lambda _ { \operatorname* { m i n } } ^ { - 1 } = \frac { 1 } { 2 } \gamma ,\tag{51}
$$

so $R ^ { 2 } \leq 1 - \textstyle { \frac { 1 } { 2 } } \eta \gamma$ and

$$
\left| \frac { r _ { + } ^ { 2 N } } { 1 - r _ { + } ^ { 2 } } - \frac { 2 ( r _ { + } r _ { - } ) ^ { N } } { 1 - r _ { + } r _ { - } } + \frac { r _ { - } ^ { 2 N } } { 1 - r _ { - } ^ { 2 } } \right| \leq \frac { 8 \left( 1 - \frac 1 2 \eta \gamma \right) ^ { N } } { \eta \gamma } .
$$

Plugging this into (41) and using $\left| \gamma ^ { 2 } - 4 \lambda ^ { - 1 } \right| = 4 \lambda ^ { - 1 } - \gamma ^ { 2 } \ge 4 \lambda ^ { - 1 } - 2 \lambda _ { \operatorname* { m a x } } ^ { - 1 } \ge 2 \lambda ^ { - 1 }$

$$
\big | \lambda ^ { \infty } - \lambda ^ { ( N ) } \big | \leq \frac { 1 6 } { \big | \gamma ^ { 2 } - 4 \lambda ^ { - 1 } \big | } \left( 1 - \frac 1 2 \eta \gamma \right) ^ { N } \leq 8 \lambda \left( 1 - \frac 1 2 \eta \gamma \right) ^ { N } .
$$

Since $1 - { \textstyle { \frac { 1 } { 2 } } } \eta \gamma \ge 0$ under (SF), the convexity inequality (49) finally gives

$$
| \lambda ^ { \infty } - \lambda ^ { ( N ) } | \leq 8 \lambda e ^ { - \eta \gamma N / 2 } .\tag{52}
$$

Conclusion. Plugging (50), respectively (52), into (40) gives

$$
\begin{array} { r l } & { W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } _ { \infty } ) \leq 8 \sqrt { \mathrm { T r } \Sigma } e ^ { - 2 \eta N / ( \lambda _ { \operatorname* { m a x } } \gamma ) } } \\ & { W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } _ { \infty } ) \leq 8 \sqrt { \mathrm { T r } \Sigma } e ^ { - \eta \gamma N / 2 } } \end{array}
$$

under (LF),

under (SF).

Remark B.11. The large- and small-friction assumptions (LF) and (SF) are stricter than conditions $\left( \mathrm { L F _ { 0 } } \right)$ and $( \mathrm { { S F _ { 0 } } ) }$ ofLemma B.9, but only through numericalfactors: their nature is the same. This stricterframework, which defines the admissible hyperparametersfor ULD, keeps the denominator $\gamma ^ { 2 } - 4 \lambda ^ { - 1 }$ bounded awayfrom zero, and affects the early-stopping and discretization bounds by a constantfactor only.

## B.3.3 Discretization error

We now bound the discretization error.

Lemma B.12. Under (LF) or (SF), the discretization error satisfies

$$
W _ { 2 } ( \mathbb { P } _ { \infty } , \mathbb { P } ) \le C \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \frac { \eta } { \gamma } ,
$$

with $\begin{array} { r } { C = \frac { 6 } { 7 } } \end{array}$ in the large-friction regime and $\begin{array} { r } { C = \frac { 3 } { 2 } } \end{array}$ in the small-friction regime.

Proof. By (11) and the inequality (37) established in the previous section,

$$
W _ { 2 } ^ { 2 } ( \mathbb { P } _ { \infty } , \mathbb { P } ) = \sum _ { i = 1 } ^ { d } \left( \frac { \lambda _ { i } ^ { \infty } - \lambda _ { i } } { \sqrt { \lambda _ { i } ^ { \infty } } + \sqrt { \lambda _ { i } } } \right) ^ { 2 } \leq \frac { 1 } { 4 } \sum _ { i = 1 } ^ { d } \frac { ( \lambda _ { i } ^ { \infty } - \lambda _ { i } ) ^ { 2 } } { \lambda _ { i } } .\tag{53}
$$

The lemma will again follow from a bound on $| \lambda ^ { \infty } - \lambda | = \lambda ^ { \infty } - \lambda .$ , for which we return to (39):

$$
\lambda ^ { \infty } - \lambda = \frac { \eta } { \gamma - \eta \lambda ^ { - 1 } } \times \frac { P ( - 1 ) + \eta \gamma } { P ( - 1 ) } = \frac { \eta } { \gamma - \eta \lambda ^ { - 1 } } \left( 1 + \frac { \eta \gamma } { P ( - 1 ) } \right) ,\tag{54}
$$

where $P ( - 1 )$ is the quantity introduced with the Jury criterion. The assumptions of Proposition B.7, stronger than those of Lemma B.9 where this criterion appears, keep $P ( - 1 )$ bounded away from zero. Indeed, in both regimes $\eta \gamma \leq 1 \colon$ this is immediate under (LF), and under (SF),

$$
\eta \gamma \leq \frac { 1 } { 2 } \lambda _ { \mathrm { m i n } } \gamma ^ { 2 } \leq \frac { 1 } { \kappa } \leq 1 .
$$

Consequently $P ( - 1 ) = 4 - 2 \eta \gamma + \eta ^ { 2 } \lambda ^ { - 1 } > 4 - 2 \eta \gamma \geq 2$ in both regimes, and (54) gives

$$
\lambda ^ { \infty } - \lambda < \frac { 3 } { 2 } \times \frac { \eta } { \gamma - \eta \lambda ^ { - 1 } } .
$$

It remains to bound $\gamma - \eta \lambda ^ { - 1 }$ from below. Under (LF),

$$
\gamma - \eta \lambda ^ { - 1 } = \gamma \left( 1 - \frac { \eta } { \gamma \lambda } \right) \geq \gamma \left( 1 - \frac { 1 } { \lambda \gamma ^ { 2 } } \right) \geq \gamma \left( 1 - \frac { \lambda _ { \operatorname* { m i n } } } { 8 \lambda } \right) \geq \frac { 7 } { 8 } \gamma ,
$$

hence

$$
\lambda ^ { \infty } - \lambda < \frac { 1 2 \eta } { 7 \gamma } .\tag{55}
$$

Under (SF), (51) gives $\gamma - \eta \lambda ^ { - 1 } \ge { \frac { 1 } { 2 } } \gamma$ , hence

$$
\lambda ^ { \infty } - \lambda < \frac { 3 \eta } { \gamma } .\tag{56}
$$

Plugging (55) and (56) into (53) completes the proof.

Proof of Proposition B.7. Combine Lemmas B.10 and B.12 through the triangle inequality (10).

## B.3.4 Choice of hyperparameters

We can now prove the first part of Theorem 1 for ULD.

ProofofTheorem 1, non-asymptotic boundfor ULD. Let $N \geq 2$ satisfy $N \geq 4 \kappa$ log N, and recall the ULD hyperparameters of Theorem 1: $\begin{array} { r } { \eta _ { N } = \sqrt { 2 \lambda _ { \operatorname* { m a x } } \kappa } \frac { \log N } { N } } \end{array}$ and $\gamma = \sqrt { 8 / \lambda _ { \mathrm { m i n } } }$ . They satisfy the large-friction assumption (LF): the only point to check is $\eta _ { N } \gamma \leq 1$ , and indeed

$$
\eta _ { N } \gamma = 4 \kappa \frac { \log N } { N } \leq 1
$$

by assumption. The first bound of Proposition B.7 therefore applies. Since $\begin{array} { r } { \frac { \eta _ { N } } { \gamma } = \frac { 1 } { 2 } \lambda _ { \mathrm { m a x } } \frac { \log N } { N } } \end{array}$ , the exponential equals $e ^ { - \log N } = 1 / N$ , and

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \frac { 8 \sqrt { \mathrm { T r } \Sigma } } { N } + \frac { 3 } { 7 } \lambda _ { \mathrm { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \frac { \log N } { N } .
$$

Using (16) once more gives the announced bound,

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \frac { \lambda _ { \operatorname* { m a x } } \sqrt { \operatorname { T r } ( \Sigma ^ { - 1 } ) } } { N } \Big ( \frac { 3 } { 7 } \log N + 8 \Big ) .
$$

The same argument in the small-friction regime (SF) gives a bound of the same order.

Corollary B.13. Let $N \geq 2 \kappa$ log N. With $\begin{array} { r } { \eta = \sqrt { 2 \lambda _ { \operatorname* { m a x } } } \frac { \log N } { N } a n d \gamma = \sqrt { 2 / \lambda _ { \operatorname* { m a x } } } , } \end{array}$

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \frac { \lambda _ { \operatorname* { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } { N } \Big ( \frac { 3 } { 2 } \log N + 8 \Big ) .
$$

Proof. With these hyperparameters, $\begin{array} { r } { \eta \gamma = 2 \frac { \log N } { N } } \end{array}$ and $\begin{array} { r } { \frac { \eta } { \gamma } = \lambda _ { \mathrm { m a x } } \frac { \log N } { N } } \end{array}$ , and the condition $N \geq$ 2κ log N is exactly $\eta \leq \frac { 1 } { 2 } \lambda _ { \operatorname* { m i n } } \gamma$ , so (SF) holds. The second bound of Proposition B.7 gives

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \le 8 \sqrt { \mathrm { T r } \Sigma } / N + \frac { 3 } { 2 } \lambda _ { \mathrm { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \frac { \log N } { N } ,
$$

and (16) concludes.

Remark B.14. To determine the optimal parameters $\eta _ { * } , \gamma _ { * } ,$ , one applies Lemma A.4 to the variable $\phi : = \eta / \gamma .$ . As for ULA, the exact minimizers are slightly different, with spectral coefficients inside the logarithms, but lead to the same bound up to numerical constants. The hyperparameters we retain are simpler: they require no knowledge ofthe spectrum ofΣ beyond its extreme eigenvalues. The same principles guided the hyperparameters ofCorollary B.13, where Lemma A.4 applies thanks to Remark B.8, which removes the variable $\gamma .$ . In whatfollows we neverthelessfocus on the regime of Theorem 1. As for ULA, the asymptotic equivalent of the next section shows that, for the chosen hyperparameters, the actual error is also oforder $\lambda _ { \operatorname* { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) }$ log N/N.

## B.3.5 Asymptotic equivalent

To complete the proof of Theorem 1, it remains to establish the asymptotic equivalent

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \underset { N  \infty } { \sim } \frac { 1 } { 4 } \lambda _ { \mathrm { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \frac { \log N } { N }
$$

for $\begin{array} { r } { \eta = \eta _ { N } = \sqrt { 2 \lambda _ { \operatorname* { m a x } } \kappa } \frac { \log N } { N } \mathrm { a n d } \gamma = \sqrt { 8 / \lambda _ { \operatorname* { m i n } } } . } \end{array}$

ProofofTheorem 1, asymptotic equivalentfor ULD. Take N large enough for the hyperparameters to satisfy (LF), so that Lemma B.9 and the bound (50) apply. Recall from (23) that

$$
\lambda ^ { \infty } = \frac { 2 \gamma ( 2 - \eta _ { N } \gamma + \eta _ { N } ^ { 2 } \lambda ^ { - 1 } ) } { ( \gamma - \eta _ { N } \lambda ^ { - 1 } ) ( 4 - 2 \eta _ { N } \gamma + \eta _ { N } ^ { 2 } \lambda ^ { - 1 } ) } \lambda .
$$

We first show that $\lambda ^ { ( N ) } \to \lambda$ . On the one hand, $\eta _ { N }  0 .$ , so

$$
\lambda ^ { \infty } \longrightarrow \frac { 4 \gamma } { 4 \gamma } \lambda = \lambda .\tag{57}
$$

On the other hand, (50) with the values of the hyperparameters gives

$$
| \lambda ^ { \infty } - \lambda ^ { ( N ) } | \leq 8 \lambda e ^ { - 2 \eta _ { N } N / ( \gamma \lambda _ { \operatorname* { m a x } } ) } = \frac { 8 \lambda } { N } ,
$$

and with (57), $\lambda ^ { ( N ) } \to \lambda$ indeed. Applying Lemma A.3 to $\mathbb { P } _ { N }$ and P then gives

$$
W _ { 2 } ^ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) = \sum _ { i = 1 } ^ { d } \Big ( \sqrt { \lambda _ { i } ^ { ( N ) } } - \sqrt { \lambda _ { i } } \Big ) ^ { 2 } = \sum _ { i = 1 } ^ { d } \left( \frac { \lambda _ { i } ^ { ( N ) } - \lambda _ { i } } { \sqrt { \lambda _ { i } ^ { ( N ) } } + \sqrt { \lambda _ { i } } } \right) ^ { 2 } \sim \frac { 1 } { 4 } \sum _ { i = 1 } ^ { d } \frac { ( \lambda _ { i } ^ { ( N ) } - \lambda _ { i } ) ^ { 2 } } { \lambda _ { i } } .\tag{58}
$$

It remains to estimate $\lambda ^ { ( N ) } - \lambda$ . By the bound above,

$$
\lambda ^ { ( N ) } - \lambda = ( \lambda ^ { ( N ) } - \lambda ^ { \infty } ) + ( \lambda ^ { \infty } - \lambda ) = \lambda ^ { \infty } - \lambda + O \left( \frac { 1 } { N } \right) ,\tag{59}
$$

and by (38),

$$
\lambda ^ { \infty } - \lambda = \frac { ( 4 - \eta _ { N } \gamma + \eta _ { N } ^ { 2 } \lambda ^ { - 1 } ) \eta _ { N } } { ( \gamma - \eta _ { N } \lambda ^ { - 1 } ) ( 4 - 2 \eta _ { N } \gamma + \eta _ { N } ^ { 2 } \lambda ^ { - 1 } ) } \underset { N \to \infty } { \sim } \frac { 4 \eta _ { N } } { 4 \gamma } = \frac { \eta _ { N } } { \gamma } = \frac { 1 } { 2 } \lambda _ { \mathrm { m a x } } \frac { \log N } { N } .
$$

Together with (59),

$$
\lambda ^ { ( N ) } - \lambda = \frac { 1 } { 2 } \lambda _ { \mathrm { m a x } } \frac { \log N } { N } + O \left( \frac { 1 } { N } \right) ,
$$

an equivalent which, as for ULA, does not depend on λ. The equivalent (58) then yields

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \underset { N  \infty } { \sim } \frac { 1 } { 4 } \lambda _ { \operatorname* { m a x } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \frac { \log N } { N } .
$$

## C Proof of Theorem 3 and Proposition 4 (diffusion sampler)

## C.1 Overview

We now turn to the proof of the following theorem, stated in Section 2.3.

Theorem 3 (Diffusion sampler). Consider an iteration budget $N \geq 1$ . Set horizon $T = ( \mathrm { T r } ( \Sigma ^ { 5 } ) / \mathrm { T r } \Sigma ) ^ { 1 / 4 } \sqrt { N } ,$ , earlystopping threshold $\delta = \mathrm { T r } \Sigma / ( d N ^ { 2 } )$ , and geometric time discretization grid $\tau _ { k } = T ( \delta / T ) ^ { k / N } f o r k \in \{ 0 , \dots , N \}$ Then the diffusion algorithm (6) yields

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \le \frac { \sqrt { \mathrm { T r } \Sigma } } { N } \Big ( \frac { 5 } { 8 } a _ { N } \rho ^ { 1 / N } \log ( \rho ^ { 2 / 5 } N ) + 1 + \sqrt { 8 } \Big ) , \quad \rho : = \frac { d \lambda _ { \operatorname* { m a x } } } { \mathrm { T r } \Sigma } \in [ 1 , d ] ,
$$

where $a _ { N } : = ( 1 + 4 N ^ { - 2 } ) ^ { 1 / 2 } N ^ { 5 / ( 2 N ) }$ satisfies $a _ { N } \leq 2 ^ { 7 / 4 }$ for all $N \geq 1$ and $a _ { N }  1$ . Moreover, this bound is asymptotically exact: under the same hyperparameters,

$$
{ \cal W } _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \underset { N  \infty } { \sim } \frac { 5 } { 8 } \sqrt { \mathrm { T r } \Sigma } \frac { \log N } { N } ,\tag{7}
$$

so that,forfixed Σ, the ratio ofthe bound to the actual error tends to 1.

Proposition 4, which differs from Theorem 3 only through the choice of hyperparameters, is restated and proved in Section C.3.

Error decomposition. The error of the diffusion algorithm of Section 2.1 combines three sources. The first is the initialization error: the process is started from $Y _ { 0 } \sim { \mathcal { N } } ( 0 , T \operatorname { I } )$ as a proxy for the true marginal $\mathbb { Q } _ { T } = \mathcal { N } ( 0 , \Sigma + T \mathbb { I } )$ which is unknown a priori. It is measured by the Wasserstein distance $W _ { 2 } ( \mathbb { P } _ { N } , \tilde { \mathbb { P } } _ { N } )$ between the law $\mathbb { P } _ { N }$ actually sampled and the law $\tilde { \mathbb { P } } _ { N }$ that would have been sampled from the exact initialization $Y _ { 0 } \sim \mathcal { N } ( 0 , \Sigma + T \mathrm { I } )$ . The second is the discretization error incurred when simulating the reverse SDE (5): it is $W _ { 2 } ( \tilde { \mathbb { P } } _ { N } , \mathbb { Q } _ { \delta } )$ , the distance, for an exact initialization, between the law produced by the discrete scheme and the law $\mathbb { Q } _ { \delta } = \mathcal { N } ( 0 , \Sigma + \delta \operatorname { I } )$ reached by the continuous process. The third is the early-stopping error $W _ { 2 } ( \mathbb { Q } _ { \delta } , \mathbb { P } )$ , due to stopping the continuous process at the threshold $\delta > 0$ . The triangle inequality formalizes this decomposition:

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq W _ { 2 } ( \mathbb { P } _ { N } , \tilde { \mathbb { P } } _ { N } ) + W _ { 2 } ( \tilde { \mathbb { P } } _ { N } , \mathbb { Q } _ { \delta } ) + W _ { 2 } ( \mathbb { Q } _ { \delta } , \mathbb { P } ) .\tag{60}
$$

Reduction to eigenvalues. All the laws in (60) are centered Gaussians with covariance in $\mathcal { A } ( \Sigma )$ : this is immediate for P and $\mathbb { Q } _ { \delta }$ , and follows from Lemma A.1 for $\mathbb { P } _ { N }$ and $\tilde { \mathbb { P } } _ { N }$ . Lemma $\mathrm { A } . 3$ therefore applies and, denoting by $\lambda _ { i } ^ { ( N ) }$ and $\tilde { \lambda } _ { i } ^ { ( N ) }$ the eigenvalues of the covariances of $\mathbb { P } _ { N }$ and $\tilde { \mathbb { P } } _ { N }$ , we record for later use

$$
W _ { 2 } ^ { 2 } ( \mathbb { P } _ { N } , \tilde { \mathbb { P } } _ { N } ) = \sum _ { i = 1 } ^ { d } \Big ( \sqrt { \lambda _ { i } ^ { ( N ) } } - \sqrt { \tilde { \lambda } _ { i } ^ { ( N ) } } \Big ) ^ { 2 } ,\tag{61}
$$

$$
W _ { 2 } ^ { 2 } ( \tilde { \mathbb { P } } _ { N } , \mathbb { Q } _ { \delta } ) = \sum _ { i = 1 } ^ { d } \Big ( \sqrt { \tilde { \lambda } _ { i } ^ { ( N ) } } - \sqrt { \lambda _ { i } + \delta } \Big ) ^ { 2 } ,\tag{62}
$$

$$
W _ { 2 } ^ { 2 } ( \mathbb { Q } _ { \delta } , \mathbb { P } ) = \sum _ { i = 1 } ^ { d } \Big ( \sqrt { \lambda _ { i } + \delta } - \sqrt { \lambda _ { i } } \Big ) ^ { 2 } .\tag{63}
$$

Outline. The proof of Theorem 3 follows the same steps as that of Theorem 1.

(i)–(ii) Parametric bound (Section C.2). Closed form of the eigenvalues $\lambda _ { i } ^ { ( N ) }$ and $\tilde { \lambda _ { i } } ^ { ( N ) }$ from the scalar recursions of Lemma A.1, then separate bounds on the three terms of (60), and summation. This gives a bound on $W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } )$ in which T and δ remain free (Proposition C.1).

(iii) Choice of hyperparameters (Section C.3). Plug the hyperparameters of Theorem 3 and Proposition 4 into the bound of Proposition C.1.

(iv) Asymptotic equivalent (Section C.4). Exact equivalent of $W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } )$ as $N  \infty$ for any geometric grid with $T \propto \sqrt { N }$ and $\delta \propto N ^ { - 2 }$ , from the closed forms of step (i). This equivalent does not depend on the proportionality constants, which shows that the spectral rate is a robust feature of the scheme.

## C.2 Parametric bound for the diffusion sampler

The goal of this section is to prove the following proposition, which bounds $W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } )$ for all values of the hyperparameters. As before, we freely drop the eigenvalue index i in the proofs.

Proposition C.1 (Parametric bound for the diffusion sampler). For every $N \geq 1$ and every $T > \delta > 0$ , with the geometric discretization $\tau _ { k } = T ( \delta / T ) ^ { k / N }$ , the error ofthe diffusion algorithm satisfies

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \frac { \sqrt { \operatorname { T r } ( ( \Sigma + \delta \operatorname { I } ) ^ { 3 } \Sigma ^ { 2 } ) } } { T ^ { 2 } } + \frac { \sqrt { \operatorname { T r } \Sigma + 4 d \delta } } { 4 N } \Big ( \frac { T } { \delta } \Big ) ^ { 1 / N } \log \frac { T } { \delta } + \sqrt { d \delta } .
$$

The three terms bound the initialization, discretization and early-stopping errors respectively.

## C.2.1 Closed form of the eigenvalues

We first establish a closed form for the eigenvalues $\lambda ^ { ( N ) }$ and $\tilde { \lambda } ^ { ( N ) }$ , valid for any discretization grid.

Lemma C.2. Fix $N \geq 1$ and consider thefollowing generalized diffusion algorithm:

$$
\begin{array} { r l } & { \Biggl \{ Y _ { 0 } \sim { \mathcal N } ( 0 , M ) , } \\ & { \Biggl \{ \forall k = 1 , \ldots , N , Y _ { k } = Y _ { k - 1 } + \eta _ { k } s _ { \tau _ { k - 1 } } ( Y _ { k - 1 } ) + \sqrt { \eta _ { k } } \xi _ { k } , } \end{array}
$$

for some $M \in { \mathcal { A } } ( \Sigma )$ , with eigenvalues $\mu _ { i } : = [ M ] _ { i }$ . Then the iterates $( Y _ { k } )$ are centered Gaussian vectors whose covariance matrices $( M _ { k } )$ belong to $\boldsymbol { \mathcal { A } } ( \Sigma )$ , and the eigenvalues $\mu _ { i } ^ { ( N ) } : = [ M _ { N } ] _ { i }$ are given by

$$
\forall i , \quad \mu _ { i } ^ { ( N ) } = \left( \frac { \lambda _ { i } + \delta } { \lambda _ { i } + T } \right) ^ { 2 } \mu _ { i } + ( \lambda _ { i } + \delta ) ^ { 2 } \sum _ { j = 1 } ^ { N } \frac { \eta _ { j } } { ( \lambda _ { i } + \tau _ { j } ) ^ { 2 } } .
$$

Proof. This generalization of (6) falls within the scope of Lemma A.1: the identification is that of Remark $\mathrm { A } . 2$ , the only change being the initial covariance M, which lies in ${ \mathcal { A } } ( \Sigma )$ by assumption. Point (ii) of the lemma gives the recursion

$$
\forall k \geq 0 , \quad \mu ^ { ( k + 1 ) } = \left( 1 - \frac { \eta _ { k + 1 } } { \lambda + \tau _ { k } } \right) ^ { 2 } \mu ^ { ( k ) } + \eta _ { k + 1 } .
$$

Since $\eta _ { k + 1 } = \tau _ { k } - \tau _ { k + 1 }$ , the contraction factor telescopes, leading to the simplification announced in Section 2.1:

$$
\mu ^ { ( k + 1 ) } = \left( \frac { \lambda + \tau _ { k } - \eta _ { k + 1 } } { \lambda + \tau _ { k } } \right) ^ { 2 } \mu ^ { ( k ) } + \eta _ { k + 1 } = \left( \frac { \lambda + \tau _ { k + 1 } } { \lambda + \tau _ { k } } \right) ^ { 2 } \mu ^ { ( k ) } + \eta _ { k + 1 } .
$$

Dividing by $( \lambda + \tau _ { k + 1 } ) ^ { 2 }$ and setting

$$
\forall j \geq 0 , \quad \nu _ { j } : = \frac { \mu ^ { ( j ) } } { ( \lambda + \tau _ { j } ) ^ { 2 } } ,
$$

we are left with the much simpler recursion

$$
\forall j \geq 0 , \quad \nu _ { j + 1 } = \nu _ { j } + \frac { \eta _ { j + 1 } } { ( \lambda + \tau _ { j + 1 } ) ^ { 2 } } , \quad \nu _ { 0 } = \frac { \mu } { ( \lambda + T ) ^ { 2 } } ,
$$

whence

$$
\nu _ { k } = \frac { \mu } { ( \lambda + T ) ^ { 2 } } + \sum _ { j = 1 } ^ { k } \frac { \eta _ { j } } { ( \lambda + \tau _ { j } ) ^ { 2 } } \quad \mathrm { a n d } \quad \mu ^ { ( k ) } = \left( \frac { \lambda + \tau _ { k } } { \lambda + T } \right) ^ { 2 } \mu + \left( \lambda + \tau _ { k } \right) ^ { 2 } \sum _ { j = 1 } ^ { k } \frac { \eta _ { j } } { ( \lambda + \tau _ { j } ) ^ { 2 } } .
$$

Taking $k = N$ , with $\tau _ { N } = \delta .$ , gives the announced formula.

Remark C.3. Applying this lemma with $M = T \mathrm { I }$ and with $M = \Sigma + T \mathrm { I }$ , both in ${ \mathcal { A } } ( \Sigma )$ , gives for every $N \geq 1$

$$
\lambda ^ { ( N ) } = \left( \frac { \lambda + \delta } { \lambda + T } \right) ^ { 2 } T + ( \lambda + \delta ) ^ { 2 } \sum _ { j = 1 } ^ { N } \frac { \eta _ { j } } { ( \lambda + \tau _ { j } ) ^ { 2 } } ,
$$

$$
\tilde { \lambda } ^ { ( N ) } = \frac { ( \lambda + \delta ) ^ { 2 } } { \lambda + T } + ( \lambda + \delta ) ^ { 2 } \sum _ { j = 1 } ^ { N } \frac { \eta _ { j } } { ( \lambda + \tau _ { j } ) ^ { 2 } } .
$$

## C.2.2 Initialization error

Lemma C.4. For every $N \geq 1$ and every $T > \delta > 0$ , the initialization error is bounded uniformly in $N .$

$$
W _ { 2 } ( \mathbb { P } _ { N } , \tilde { \mathbb { P } } _ { N } ) \leq \frac { \sqrt { \mathrm { T r } ( ( \Sigma + \delta \operatorname { I } ) ^ { 3 } \Sigma ^ { 2 } ) } } { T ^ { 2 } } .
$$

Proof. Fix $N \geq 1$ and $T > \delta > 0$ throughout. By (61),

$$
W _ { 2 } ^ { 2 } ( \mathbb { P } _ { N } , \tilde { \mathbb { P } } _ { N } ) = \sum _ { i = 1 } ^ { d } \Big ( \sqrt { \lambda _ { i } ^ { ( N ) } } - \sqrt { \tilde { \lambda } _ { i } ^ { ( N ) } } \Big ) ^ { 2 } .\tag{64}
$$

Starting from the expression of $\tilde { \lambda } ^ { ( N ) }$ in Remark C.3, and using $\begin{array} { r } { ( \lambda + \delta ) ^ { 2 } \cdot \frac { 1 } { \lambda + \delta } = \lambda + \delta . } \end{array}$

$$
\begin{array} { l } { \displaystyle \tilde { \lambda } ^ { ( N ) } = ( \lambda + \delta ) ^ { 2 } \Bigg ( \sum _ { j = 1 } ^ { N } \frac { \eta _ { j } } { ( \lambda + \tau _ { j } ) ^ { 2 } } + \frac { 1 } { \lambda + T } \Bigg ) } \\ { \displaystyle = \lambda + \delta + ( \lambda + \delta ) ^ { 2 } \Bigg ( \sum _ { j = 1 } ^ { N } \frac { \eta _ { j } } { ( \lambda + \tau _ { j } ) ^ { 2 } } - \left( \frac { 1 } { \lambda + \delta } - \frac { 1 } { \lambda + T } \right) \Bigg ) = \lambda + \delta + ( \lambda + \delta ) ^ { 2 } \Delta , } \end{array}\tag{65}
$$

where

$$
\Delta : = \sum _ { j = 1 } ^ { N } \frac { \eta _ { j } } { ( \lambda + \tau _ { j } ) ^ { 2 } } - \left( \frac { 1 } { \lambda + \delta } - \frac { 1 } { \lambda + T } \right) = \sum _ { j = 1 } ^ { N } \frac { \eta _ { j } } { ( \lambda + \tau _ { j } ) ^ { 2 } } - \int _ { \delta } ^ { T } \frac { \mathrm { d } \tau } { ( \lambda + \tau ) ^ { 2 } } .
$$

$\Delta$ is the quadrature error made when approximating $\begin{array} { r } { \int _ { \delta } ^ { T } ( \lambda + \tau ) ^ { - 2 } \mathrm { d } \tau } \end{array}$ by the Riemann sum $\textstyle \sum _ { j = 1 } ^ { N } \eta _ { j } ( \lambda + \tau _ { j } ) ^ { - 2 }$ , in which the decreasing integrand is evaluated at the left endpoint of each interval $[ \tau _ { j } , \tau _ { j - 1 } ] ;$ ; it is therefore positive. Explicitly, since $\tau _ { 0 } = T$ and $\tau _ { N } = \delta$

$$
\begin{array} { l } { \displaystyle \Delta = \sum _ { j = 1 } ^ { N } \frac { \eta _ { j } } { ( \lambda + \tau _ { j } ) ^ { 2 } } - \sum _ { j = 1 } ^ { N } \left( \frac { 1 } { \lambda + \tau _ { j } } - \frac { 1 } { \lambda + \tau _ { j - 1 } } \right) } \\ { \displaystyle = \sum _ { j = 1 } ^ { N } \left( \frac { \eta _ { j } } { ( \lambda + \tau _ { j } ) ^ { 2 } } - \frac { \eta _ { j } } { ( \lambda + \tau _ { j } ) ( \lambda + \tau _ { j - 1 } ) } \right) } \\ { \displaystyle = \sum _ { j = 1 } ^ { N } \frac { \eta _ { j } ^ { 2 } } { ( \lambda + \tau _ { j } ) ^ { 2 } ( \lambda + \tau _ { j - 1 } ) } > 0 , } \end{array}\tag{66}
$$

where we used $\tau _ { j - 1 } - \tau _ { j } = \eta _ { j }$ twice. With (65), this gives

$$
\tilde { \lambda } ^ { ( N ) } > \lambda + \delta .\tag{67}
$$

Remark C.3 also gives the difference of the two eigenvalues,

$$
\tilde { \lambda } ^ { ( N ) } - \lambda ^ { ( N ) } = \left( \frac { \lambda + \delta } { \lambda + T } \right) ^ { 2 } \lambda ,\tag{68}
$$

so that, by (67),

$$
\sqrt { \tilde { \lambda } ^ { ( N ) } } - \sqrt { \lambda ^ { ( N ) } } = \frac { \tilde { \lambda } ^ { ( N ) } - \lambda ^ { ( N ) } } { \sqrt { \tilde { \lambda } ^ { ( N ) } } + \sqrt { \lambda ^ { ( N ) } } } \leq \frac { \tilde { \lambda } ^ { ( N ) } - \lambda ^ { ( N ) } } { \sqrt { \lambda + \delta } } = \frac { ( \lambda + \delta ) ^ { 3 / 2 } \lambda } { ( \lambda + T ) ^ { 2 } } \leq \frac { ( \lambda + \delta ) ^ { 3 / 2 } \lambda } { T ^ { 2 } } .
$$

Squaring, summing over i and using (64) gives the bound of the lemma.

## C.2.3 Discretization error

Lemma C.5. For every $N \geq 1$ and every $T > \delta > 0$ , with the geometric discretization $\tau _ { k } = T ( \delta / T ) ^ { k / N }$ , the discretization error satisfies

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { Q } _ { \delta } ) \leq \frac { \sqrt { \mathrm { T r } \Sigma + 4 d \delta } } { 4 N } \Big ( \frac { T } { \delta } \Big ) ^ { 1 / N } \log \frac { T } { \delta } .
$$

Proof. Fix $N \geq 1$ and $T > \delta > 0$ . By (62) and (67),

$$
\begin{array} { l } { { W _ { 2 } ^ { 2 } \big ( \tilde { \mathbb { P } } _ { N } , \mathbb { Q } _ { \delta } \big ) = \displaystyle \sum _ { i = 1 } ^ { d } \left( \sqrt { \tilde { \lambda } _ { i } ^ { ( N ) } } - \sqrt { \lambda _ { i } + \delta } \right) ^ { 2 } = \displaystyle \sum _ { i = 1 } ^ { d } \left( \frac { \tilde { \lambda } _ { i } ^ { ( N ) } - ( \lambda _ { i } + \delta ) } { \sqrt { \tilde { \lambda } _ { i } ^ { ( N ) } } + \sqrt { \lambda _ { i } + \delta } } \right) ^ { 2 } } } \\ { { \displaystyle \qquad \leq \frac { 1 } { 4 } \sum _ { i = 1 } ^ { d } \frac { \big ( \tilde { \lambda } _ { i } ^ { ( N ) } - ( \lambda _ { i } + \delta ) \big ) ^ { 2 } } { \lambda _ { i } + \delta } } , } \end{array}
$$

and with (65),

$$
W _ { 2 } ^ { 2 } ( \widetilde { \mathbb { P } } _ { N } , \mathbb { Q } _ { \delta } ) \leq \frac { 1 } { 4 } \sum _ { i = 1 } ^ { d } ( \lambda _ { i } + \delta ) ^ { 3 } \Delta _ { i } ^ { 2 } .\tag{69}
$$

It thus remains to bound $\Delta$ for the geometric grid $\tau _ { k } = T ( \delta / T ) ^ { k / N }$ . Let $q : = ( \delta / T ) ^ { 1 / N }$ denote its ratio, so that $\tau _ { k } = q \tau _ { k - }$ and

$$
\forall k \geq 1 , \quad \eta _ { k } = ( q ^ { - 1 } - 1 ) \tau _ { k } .
$$

Returning to (66),

$$
\Delta = ( q ^ { - 1 } - 1 ) \sum _ { j = 1 } ^ { N } \frac { \tau _ { j } } { \lambda + \tau _ { j } } \left( \frac { 1 } { \lambda + \tau _ { j } } - \frac { 1 } { \lambda + \tau _ { j - 1 } } \right)\tag{70}
$$

$$
= ( q ^ { - 1 } - 1 ) \sum _ { j = 1 } ^ { N } f ( \tau _ { j } ) \bigl ( G ( \tau _ { j } ) - G ( \tau _ { j - 1 } ) \bigr ) ,\tag{71}
$$

where

$$
f : \tau \longmapsto { \frac { \tau } { \lambda + \tau } } , \quad G : \tau \longmapsto { \frac { 1 } { \lambda + \tau } } .
$$

Since $\tau _ { j } < \tau _ { j - 1 }$ and $f$ is increasing,

$$
f ( \tau _ { j } ) \big ( G ( \tau _ { j } ) - G ( \tau _ { j - 1 } ) \big ) = f ( \tau _ { j } ) \int _ { \tau _ { j } } ^ { \tau _ { j - 1 } } \frac { \mathrm { d } \tau } { ( \lambda + \tau ) ^ { 2 } } \leq \int _ { \tau _ { j } } ^ { \tau _ { j - 1 } } \frac { f ( \tau ) \mathrm { d } \tau } { ( \lambda + \tau ) ^ { 2 } } = \int _ { \tau _ { j } } ^ { \tau _ { j - 1 } } \frac { \tau \mathrm { d } \tau } { ( \lambda + \tau ) ^ { 3 } } ,
$$

and summing over $j$ in (71),

$$
\Delta \leq ( q ^ { - 1 } - 1 ) \int _ { \delta } ^ { T } \frac { \tau \mathrm { d } \tau } { ( \lambda + \tau ) ^ { 3 } } .\tag{72}
$$

One checks that $\begin{array} { r } { \tau \mapsto \frac { \lambda } { 2 ( \lambda + \tau ) ^ { 2 } } - \frac { 1 } { \lambda + \tau } } \end{array}$ is an antiderivative of the integrand, hence

$$
\begin{array} { r l r } {  { \Delta \le ( q ^ { - 1 } - 1 ) [ \frac { \lambda } { 2 ( \lambda + \tau ) ^ { 2 } } - \frac { 1 } { \lambda + \tau } ] _ { \delta } ^ { T } = ( q ^ { - 1 } - 1 ) ( \frac { \lambda + 2 \delta } { 2 ( \lambda + \delta ) ^ { 2 } } - \frac { \lambda + 2 T } { 2 ( \lambda + T ) ^ { 2 } } ) } } \\ & { } & { \le ( q ^ { - 1 } - 1 ) \frac { \lambda + 2 \delta } { 2 ( \lambda + \delta ) ^ { 2 } } , } \end{array}
$$

and therefore

$$
( \lambda + \delta ) ^ { 3 } \Delta ^ { 2 } \leq \frac { 1 } { 4 } ( q ^ { - 1 } - 1 ) ^ { 2 } \frac { ( \lambda + 2 \delta ) ^ { 2 } } { \lambda + \delta } \leq \frac { 1 } { 4 } ( q ^ { - 1 } - 1 ) ^ { 2 } ( \lambda + 4 \delta ) ,
$$

where the last inequality is $( \lambda + 2 \delta ) ^ { 2 } \leq ( \lambda + 2 \delta ) ^ { 2 } + \lambda \delta = ( \lambda + \delta ) ( \lambda + 4 \delta )$ . Plugging this into (69),

$$
W _ { 2 } ( \widetilde { \mathbb { P } } _ { N } , \mathbb { Q } _ { \delta } ) \leq \frac { 1 } { 4 } ( q ^ { - 1 } - 1 ) \sqrt { \mathrm { T r } \Sigma + 4 d \delta } .\tag{73}
$$

Finally, the inequality $e ^ { u } - 1 \leq u e ^ { u }$ , valid for all $u \in \mathbb { R }$ , applied to $\textstyle u = { \frac { 1 } { N } }$ log $\textstyle { \frac { T } { \delta } }$ gives

$$
q ^ { - 1 } - 1 = \left( \frac { T } { \delta } \right) ^ { 1 / N } - 1 \leq \frac { 1 } { N } \Big ( \frac { T } { \delta } \Big ) ^ { 1 / N } \log \frac { T } { \delta } ,
$$

which together with (73) concludes the proof.

## C.2.4 Early-stopping error

Lemma C.6. For every $\delta > 0$ , the early-stopping error satisfies

$$
W _ { 2 } ( \mathbb { Q } _ { \delta } , \mathbb { P } ) \leq { \sqrt { d \delta } } .
$$

Proof. By (63), it suffices to note that for every $\lambda > 0$

$$
( \sqrt { \lambda + \delta } - \sqrt { \lambda } ) ^ { 2 } = 2 \lambda + \delta - 2 \sqrt { \lambda } \sqrt { \lambda + \delta } < 2 \lambda + \delta - 2 \lambda = \delta .
$$

Proof of Proposition C.1. Combine Lemmas C.4, C.5 and C.6 through the triangle inequality (60).

## C.3 Choice of hyperparameters

Thanks to Proposition C.1, we can now prove the non-asymptotic bounds of Theorem 3 (including the properties of $\left( a _ { N } \right)$ stated there) and of Proposition 4.

Proof of Theorem $^ { 3 , }$ non-asymptotic bound. Let $N \geq 1$ and recall the hyperparameters of Theorem $3 { \mathrm { : } }$

$$
T = \left( { \frac { \mathrm { T r } ( \Sigma ^ { 5 } ) } { \mathrm { T r } \Sigma } } \right) ^ { 1 / 4 } \sqrt { N } , \quad \delta = { \frac { \mathrm { T r } \Sigma } { d N ^ { 2 } } } .
$$

We first check that $T > \delta ,$ , as required by Proposition C.1. By Holder’s inequality,¨ $\mathrm { T r } ( \Sigma ^ { 5 } ) \geq ( \mathrm { T r } \Sigma ) ^ { 5 } / d ^ { 4 }$ , hence

$$
T \geq \left( { \frac { ( { \mathrm { T r } } \Sigma ) ^ { 4 } } { d ^ { 4 } } } \right) ^ { 1 / 4 } { \sqrt { N } } = { \frac { { \mathrm { T r } } \Sigma } { d } } { \sqrt { N } } ,
$$

which exceeds $\begin{array} { r } { \delta = { \frac { { \mathrm { T r } } \Sigma } { d } } N ^ { - 2 } } \end{array}$ as soon as $N \geq 2 .$ . For $N = 1$ , the inequality $T > \delta$ still holds unless Holder’s¨ inequality is an equality, i.e., unless $\Sigma$ is a scalar matrix. In that isotropic case one has $T = \delta .$ , the algorithm performs a single step of size $\eta _ { 1 } = 0$ , and $\mathbb { P } _ { 1 } = \mathcal { N } ( 0 , T \mathbb { I } ) = \mathbb { P }$ , so that $W _ { 2 } ( \mathbb { P } _ { 1 } , \mathbb { P } ) = 0$ and the bound holds trivially. This degenerate case is of no interest, but covering it allows Theorem 3 to be stated for all $N \geq 1$ . We assume $T > \delta$ from now on.

Proposition C.1 then gives

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \frac { \sqrt { \operatorname { T r } ( ( \Sigma + \delta \operatorname { I } ) ^ { 3 } \Sigma ^ { 2 } ) } } { T ^ { 2 } } + \frac { \sqrt { \operatorname { T r } \Sigma + 4 d \delta } } { 4 N } \Big ( \frac { T } { \delta } \Big ) ^ { 1 / N } \log \frac { T } { \delta } + \sqrt { d \delta } ,\tag{74}
$$

and we bound the three terms in turn.

Initialization. With the value of $T _ { \cdot }$

$$
{ \frac { { \sqrt { \operatorname { T r } ( ( \Sigma + \delta \operatorname { I } ) ^ { 3 } \Sigma ^ { 2 } ) } } } { T ^ { 2 } } } = { \frac { \sqrt { \operatorname { T r } \Sigma } } { N } } \times { \sqrt { \frac { \operatorname { T r } ( ( \Sigma + \delta \operatorname { I } ) ^ { 3 } \Sigma ^ { 2 } ) } { \operatorname { T r } ( \Sigma ^ { 5 } ) } } } .\tag{75}
$$

Expanding the binomial and inserting the value of $\delta ,$

$$
\mathrm { T r } ( ( \Sigma + \delta \operatorname { I } ) ^ { 3 } \Sigma ^ { 2 } ) = \sum _ { j = 0 } ^ { 3 } { \binom { 3 } { j } } \delta ^ { j } \mathrm { T r } ( \Sigma ^ { 5 - j } ) = \sum _ { j = 0 } ^ { 3 } { \binom { 3 } { j } } \left( \frac { \mathrm { T r } \Sigma } { d N ^ { 2 } } \right) ^ { j } \mathrm { T r } ( \Sigma ^ { 5 - j } ) .
$$

By Holder’s inequality again,¨ $\mathrm { T r } ( \Sigma ^ { k } ) \leq d ^ { 1 - k / 5 } \mathrm { T r } ( \Sigma ^ { 5 } ) ^ { k / 5 } \mathrm { f o r } 1 \leq k \leq 5 ;$ applied to $k = 1$ and $k = 5 - j$ , this gives $( \mathrm { { { \hat { T r } } } } \Sigma / d ) ^ { j } \leq d ^ { - j / 5 } \mathrm { { \hat { T r } } } ( \bar { \Sigma ^ { 5 } } ) ^ { j / 5 }$ and $\operatorname { T r } ( \Sigma ^ { 5 - j } ) \leq d ^ { j / 5 } \operatorname { T r } ( \Sigma ^ { 5 } ) ^ { 1 - j / 5 }$ , so the powers of d cancel and

$$
\operatorname { T r } ( ( \Sigma + \delta \operatorname { I } ) ^ { 3 } \Sigma ^ { 2 } ) \leq \sum _ { j = 0 } ^ { 3 } { \binom { 3 } { j } } { \frac { 1 } { N ^ { 2 j } } } \operatorname { T r } ( \Sigma ^ { 5 } ) = \left( 1 + { \frac { 1 } { N ^ { 2 } } } \right) ^ { 3 } \operatorname { T r } ( \Sigma ^ { 5 } ) .
$$

Returning to (75),

$$
{ \frac { { \sqrt { \operatorname { T r } ( ( \Sigma + \delta \operatorname { I } ) ^ { 3 } \Sigma ^ { 2 } ) } } } { T ^ { 2 } } } \leq \left( 1 + { \frac { 1 } { N ^ { 2 } } } \right) ^ { 3 / 2 } { \frac { \sqrt { \operatorname { T r } \Sigma } } { N } } \leq { \sqrt { 8 } } { \frac { \sqrt { \operatorname { T r } \Sigma } } { N } } .
$$

Discretization. On the one hand, $\mathrm { T r } \Sigma + 4 d \delta = ( 1 + 4 N ^ { - 2 } ) \mathrm { T r } \Sigma$ . On the other hand,

$$
\frac { T } { \delta } = \left( \frac { \mathrm { T r } ( \Sigma ^ { 5 } ) } { \mathrm { T r } \Sigma } \right) ^ { 1 / 4 } \frac { d N ^ { 5 / 2 } } { \mathrm { T r } \Sigma } \leq \frac { d \lambda _ { \mathrm { m a x } } } { \mathrm { T r } \Sigma } N ^ { 5 / 2 } = \rho N ^ { 5 / 2 } ,
$$

where we used $\mathrm { T r } ( \Sigma ^ { 5 } ) \le \lambda _ { \mathrm { m a x } } ^ { 4 } \mathrm { T r } \Sigma$ , and where $\rho : = d \lambda _ { \operatorname* { m a x } } / \operatorname { T r } \Sigma \in [ 1 , d ]$ is the ratio between the largest eigenvalue of Σ and the arithmetic mean of its spectrum: the more anisotropic Σ, the larger $\rho .$ Since $x \mapsto x ^ { 1 / N }$ log x is increasing on $\lbrack 1 , \infty )$ and $T / \delta > 1$ , the discretization term is at most

$$
\frac { ( 1 + 4 N ^ { - 2 } ) ^ { 1 / 2 } \sqrt { \mathrm { T r } \Sigma } } { 4 N } ( \rho N ^ { 5 / 2 } ) ^ { 1 / N } \log ( \rho N ^ { 5 / 2 } ) = \frac { 5 } { 8 } a _ { N } \frac { \sqrt { \mathrm { T r } \Sigma } } { N } \rho ^ { 1 / N } \log ( \rho ^ { 2 / 5 } N ) ,
$$

where we wrote $\log ( \rho N ^ { 5 / 2 } ) = \textstyle { \frac { 5 } { 2 } } \log ( \rho ^ { 2 / 5 } N )$ and $a _ { N } : = ( 1 + 4 N ^ { - 2 } ) ^ { 1 / 2 } N ^ { 5 / ( 2 N ) }$ as in the statement.

Early stopping. Immediately, $\sqrt { d \delta } = \sqrt { \mathrm { T r } \Sigma } / N$

Summing the three bounds in (74) gives the bound of Theorem 3,

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \le \frac { \sqrt { \mathrm { T r } \Sigma } } { N } \Big ( \frac 5 8 a _ { N } \rho ^ { 1 / N } \log ( \rho ^ { 2 / 5 } N ) + 1 + \sqrt { 8 } \Big ) .
$$

Properties $o f \left( a _ { N } \right)$ . Both factors of $a N$ tend to 1, so $a _ { N }  1$ . Moreover, $( 1 + 4 N ^ { - 2 } ) ^ { 1 / 2 }$ is decreasing in $N .$ , and $N ^ { 1 / N }$ is decreasing for $N \geq 3$ (as the study of $x \mapsto x ^ { 1 / x }$ shows), so $\left( a _ { N } \right) _ { N \geq 3 }$ is decreasing and

$$
\operatorname* { s u p } _ { N > 1 } a _ { N } = \operatorname* { m a x } \left( a _ { 1 } , a _ { 2 } , a _ { 3 } \right) = \operatorname* { m a x } \left( { \sqrt { 5 } } , { \sqrt { 2 } } \times 2 ^ { 5 / 4 } , { \frac { 1 } { 3 } } { \sqrt { 1 3 } } \times 3 ^ { 5 / 6 } \right) = 2 ^ { 7 / 4 } .\tag{76}
$$

In particular $\textstyle { \frac { 5 } { 8 } } a _ { N } \leq 5 / 2 ^ { 5 / 4 }$ , which yields a version of the bound with fixed numerical constants.

We now turn to Proposition 4, which we restate for convenience.

Proposition 4. With the alternative choice $T = \lambda _ { \operatorname* { m a x } } \sqrt { N } ,$ , early-stopping threshold $\delta = \lambda _ { \operatorname* { m a x } } / N ^ { 2 }$ , and geometric time discretization grid $\tau _ { k } = T ( \delta / T ) ^ { k / N } f o r k \in \{ 0 , \dots , N \}$ , the diffusion algorithm (6) satisfies, for every $N \geq 1$

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \frac { \sqrt { d \lambda _ { \operatorname* { m a x } } } } { N } \biggr ( \frac { 5 } { 2 ^ { 5 / 4 } } \log N + \Bigl ( \frac { 5 } { 4 } \Bigr ) ^ { 3 / 2 } + 1 \biggr ) .
$$

Proof. Let $N \geq 1$ and set $T = \lambda _ { \operatorname* { m a x } } \sqrt { N }$ and $\delta = \lambda _ { \operatorname* { m a x } } / N ^ { 2 }$ as in the statement. The condition $T > \delta$ of Proposition C.1 holds as soon as $N \geq 2$ . For $N = 1$ , we have $T = \delta ;$ as in the proof of Theorem 3, the algorithm performs a single step of size $\eta _ { 1 } = 0$ , and $\mathbb { P } _ { 1 } = \mathcal { N } ( 0 , \lambda _ { \operatorname* { m a x } } \mathbb { I } )$ . Hence, $W _ { 2 } ( \mathbb { P } _ { 1 } , \mathbb { P } ) \leq \sqrt { d \lambda _ { \operatorname* { m a x } } }$ and the bound of Proposition 4 holds. As before, this degenerate case only serves to state the proposition for all $N \geq 1$

From now on, we assume $N \geq 2 .$ , so that $T > \delta$ and Proposition C.1 applies:

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \frac { \sqrt { \operatorname { T r } ( ( \Sigma + \delta \operatorname { I } ) ^ { 3 } \Sigma ^ { 2 } ) } } { T ^ { 2 } } + \frac { \sqrt { \operatorname { T r } \Sigma + 4 d \delta } } { 4 N } \Big ( \frac { T } { \delta } \Big ) ^ { 1 / N } \log \frac { T } { \delta } + \sqrt { d \delta } .\tag{77}
$$

We bound the three terms in turn.

Initialization. Bounding each eigenvalue $\lambda _ { i }$ by $\lambda _ { \operatorname* { m a x } } .$ , we have

$$
\frac { \sqrt { \mathrm { T r } ( ( \Sigma + \delta \mathrm { I } ) ^ { 3 } \Sigma ^ { 2 } ) } } { T ^ { 2 } } \le \frac { \sqrt { ( 1 + N ^ { - 2 } ) ^ { 3 } d \lambda _ { \operatorname* { m a x } } ^ { 5 } } } { \lambda _ { \operatorname* { m a x } } ^ { 2 } N } = ( 1 + N ^ { - 2 } ) ^ { 3 / 2 } \frac { \sqrt { d \lambda _ { \operatorname* { m a x } } } } { N } .
$$

Since $N \geq 2$ , we get

$$
\frac { \sqrt { \mathrm { T r } ( ( \Sigma + \delta \mathrm { I } ) ^ { 3 } \Sigma ^ { 2 } ) } } { T ^ { 2 } } \le \Big ( \frac { 5 } { 4 } \Big ) ^ { 3 / 2 } \frac { \sqrt { d \lambda _ { \operatorname* { m a x } } } } { N } .
$$

Discretization. Since $T / \delta = N ^ { 5 / 2 }$ , using $\operatorname { T r } \Sigma \leq d \lambda _ { \operatorname* { m a x } }$ gives

$$
\begin{array} { r l } & { \frac { \sqrt { \mathrm { T r } \Sigma + 4 d \delta } } { 4 N } \Big ( \frac { T } { \delta } \Big ) ^ { 1 / N } \log \frac { T } { \delta } \leq \frac { \sqrt { d \lambda _ { \operatorname* { m a x } } ( 1 + 4 N ^ { - 2 } ) } } { 4 N } N ^ { 5 / ( 2 N ) } \log ( N ^ { 5 / 2 } ) } \\ & { \qquad = \frac { 5 } { 8 } ( 1 + 4 N ^ { - 2 } ) ^ { 1 / 2 } N ^ { 5 / ( 2 N ) } \sqrt { d \lambda _ { \operatorname* { m a x } } } \frac { \log N } { N } } \\ & { \qquad = \frac { 5 } { 8 } a _ { N } \sqrt { d \lambda _ { \operatorname* { m a x } } } \frac { \log N } { N } } \\ & { \qquad \leq \frac { 5 } { 2 ^ { 5 / 4 } } \sqrt { d \lambda _ { \operatorname* { m a x } } } \frac { \log N } { N } , } \end{array}
$$

where the last inequality is (76).

Early stopping. Immediately, $\sqrt { d \delta } = \sqrt { d \lambda _ { \operatorname* { m a x } } } / N$

Summing the three bounds in (77) then yields the announced bound.

The equivalent (7), on which the asymptotic exactness claimed in Theorem 3 rests, is the object of the next section.

## C.4 Asymptotic equivalent

We prove the following proposition, of which the asymptotic equivalent of Theorem 3 is a special case. It also covers the hyperparameters of Proposition 4, namely $A = B = \lambda _ { \operatorname* { m a x } } .$ , so that the asymptotic equivalent (7) holds in that case as well.

Proposition C.7. Let $A , B > 0$ be constants independent of N. With hyperparameters of the form

$$
T = A \sqrt N , \quad \delta = \frac { B } { N ^ { 2 } } ,
$$

and the geometric discretization $\tau _ { k } = T ( \delta / T ) ^ { k / N }$ , the error of the diffusion algorithm satisfies

$$
{ \cal W } _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \underset { N  \infty } { \sim } \frac { 5 } { 8 } \sqrt { \mathrm { T r } \Sigma } \frac { \log N } { N } .
$$

Proof. Let $T = A \sqrt { N } , \delta = B N ^ { - 2 }$ and $\tau _ { k } = T ( \delta / T ) ^ { k / N }$ as in the statement, and take N large enough that $T > \delta$ By Lemma A.3,

$$
W _ { 2 } ^ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) = \sum _ { i = 1 } ^ { d } \Big ( \sqrt { \lambda _ { i } ^ { ( N ) } } - \sqrt { \lambda _ { i } } \Big ) ^ { 2 } = \sum _ { i = 1 } ^ { d } \Bigg ( \frac { \lambda _ { i } ^ { ( N ) } - \lambda _ { i } } { \sqrt { \lambda _ { i } ^ { ( N ) } } + \sqrt { \lambda _ { i } } } \Bigg ) ^ { 2 } ,\tag{78}
$$

so we need an equivalent of $\lambda ^ { ( N ) } - \lambda .$ , which we split as

$$
\lambda ^ { ( N ) } - \lambda = ( \lambda ^ { ( N ) } - \tilde { \lambda } ^ { ( N ) } ) + ( \tilde { \lambda } ^ { ( N ) } - ( \lambda + \delta ) ) + \delta .
$$

By (68), the first term is $\begin{array} { r } { - \left( \frac { \lambda + \delta } { \lambda + T } \right) ^ { 2 } \lambda = O ( T ^ { - 2 } ) = O ( N ^ { - 1 } ) } \end{array}$ , and the third is $\delta = { \cal O } ( N ^ { - 2 } )$ . Hence

$$
\lambda ^ { ( N ) } - \lambda = \tilde { \lambda } ^ { ( N ) } - ( \lambda + \delta ) + O \left( \frac { 1 } { N } \right) ,\tag{79}
$$

and by (65) the remaining term is

$$
\begin{array} { r } { \tilde { \lambda } ^ { ( N ) } - ( \lambda + \delta ) = ( \lambda + \delta ) ^ { 2 } \Delta . } \end{array}\tag{80}
$$

Everything thus reduces to an equivalent of the quadrature error $\Delta$ , which we obtain by squeezing it between two quantities with the same asymptotics.

Upper bound. By (72), since the integrand is positive,

$$
\begin{array} { l } { \Delta \le ( q ^ { - 1 } - 1 ) \displaystyle \int _ { \delta } ^ { T } \frac { \tau \mathrm { d } \tau } { ( \lambda + \tau ) ^ { 3 } } \le ( q ^ { - 1 } - 1 ) \displaystyle \int _ { 0 } ^ { \infty } \frac { \tau \mathrm { d } \tau } { ( \lambda + \tau ) ^ { 3 } } } \\ { = ( q ^ { - 1 } - 1 ) \displaystyle \left[ \frac { \lambda } { 2 ( \lambda + \tau ) ^ { 2 } } - \frac { 1 } { \lambda + \tau } \right] _ { 0 } ^ { \infty } = \frac { q ^ { - 1 } - 1 } { 2 \lambda } , } \end{array}
$$

and, since $\textstyle { \frac { 1 } { N } } \log { \frac { A N ^ { 5 / 2 } } { B } } \to 0$

$$
q ^ { - 1 } - 1 = \left( \frac { A N ^ { 5 / 2 } } { B } \right) ^ { 1 / N } - 1 = \exp \left( \frac { 1 } { N } \log \frac { A N ^ { 5 / 2 } } { B } \right) - 1 \sim \frac { 1 } { N } \log \frac { A N ^ { 5 / 2 } } { B } \sim \frac { 5 } { 2 } \frac { \log N } { N } .\tag{81}
$$

Lower bound. Since $q \leq 1$ and $\tau _ { j } = q \tau _ { j - 1 }$ , for every $j \geq 1$

$$
f ( \tau _ { j } ) = \frac { q \tau _ { j - 1 } } { \lambda + q \tau _ { j - 1 } } \geq \frac { q \tau _ { j - 1 } } { \lambda + \tau _ { j - 1 } } = q f ( \tau _ { j - 1 } ) ,
$$

where $\begin{array} { r } { f : \tau \mapsto \frac { \tau } { \lambda + \tau } } \end{array}$ is the increasing function of Section C.2. Returning to (70) and using $( q ^ { - 1 } - 1 ) q = 1 - q$

$$
\begin{array} { l } { \displaystyle \Delta \geq ( 1 - q ) \sum _ { j = 1 } ^ { N } f ( \tau _ { j - 1 } ) \left( \frac { 1 } { \lambda + \tau _ { j } } - \frac { 1 } { \lambda + \tau _ { j - 1 } } \right) = ( 1 - q ) \sum _ { j = 1 } ^ { N } f ( \tau _ { j - 1 } ) \int _ { \tau _ { j } } ^ { \tau _ { j - 1 } } \frac { \mathrm { d } \tau } { ( \lambda + \tau ) ^ { 2 } } } \\ { \displaystyle \quad \geq ( 1 - q ) \sum _ { j = 1 } ^ { N } \int _ { \tau _ { j } } ^ { \tau _ { j - 1 } } \frac { f ( \tau ) \mathrm { d } \tau } { ( \lambda + \tau ) ^ { 2 } } = ( 1 - q ) \int _ { \delta } ^ { T } \frac { \tau \mathrm { d } \tau } { ( \lambda + \tau ) ^ { 3 } } , } \end{array}
$$

the second inequality using $f ( \tau ) \leq f ( \tau _ { j - 1 } ) \mathrm { o n } \left[ \tau _ { j } , \tau _ { j - 1 } \right]$ . As $\delta \to 0$ and $T \to \infty$ , the integrability of $\tau \mapsto \tau ( \lambda { + } \tau )$ −3 gives

$$
\int _ { \delta } ^ { T } \frac { \tau \mathrm { d } \tau } { ( \lambda + \tau ) ^ { 3 } } \underset { N  \infty } { \longrightarrow } \int _ { 0 } ^ { \infty } \frac { \tau \mathrm { d } \tau } { ( \lambda + \tau ) ^ { 3 } } = \frac { 1 } { 2 \lambda } ,
$$

while $q \to 1$ and (81) give $\begin{array} { r } { 1 - q = q ( q ^ { - 1 } - 1 ) \sim \frac { 5 } { 2 } \frac { \log N } { N } } \end{array}$

Conclusion. We have shown

$$
\underbrace { ( 1 - q ) \int _ { \delta } ^ { T } \frac { \tau \mathrm { d } \tau } { ( \lambda + \tau ) ^ { 3 } } } _ { \sim \frac { 5 } { 4 \lambda } \frac { \log N } { N } } \le \Delta \le \underbrace { \frac { q ^ { - 1 } - 1 } { 2 \lambda } } _ { \sim \frac { 5 } { 4 \lambda } \frac { \log N } { N } } , \quad \mathrm { h e n c e } \quad \Delta \underset { N  \infty } { \sim } \frac { 5 } { 4 \lambda } \frac { \log N } { N } .
$$

Plugging this into (80) and then (79), and using $\delta \to 0$

$$
\lambda ^ { ( N ) } - \lambda \sim \tilde { \lambda } ^ { ( N ) } - ( \lambda + \delta ) \sim \frac { 5 \lambda } { 4 } \frac { \log N } { N } .
$$

In particular $\lambda ^ { ( N ) } \to \lambda$ , so the denominators in (78) tend to $2 \sqrt { \lambda }$ and

$$
W _ { 2 } ^ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \sim \sum _ { i = 1 } ^ { d } \frac { ( \lambda _ { i } ^ { ( N ) } - \lambda _ { i } ) ^ { 2 } } { 4 \lambda _ { i } } \sim \left( \frac { 5 } { 8 } \frac { \log N } { N } \right) ^ { 2 } \sum _ { i = 1 } ^ { d } \lambda _ { i } = \left( \frac { 5 } { 8 } \frac { \log N } { N } \right) ^ { 2 } \operatorname { T r } \Sigma ,
$$

which is the announced equivalent.

Proof of Theorem 3, asymptotic equivalent and exactness. The hyperparameters of Theorem 3 are of the form of Proposition C.7 with $A = ( \mathrm { T r } ( \bar { \Sigma } ^ { 5 } ) / \mathrm { T r } \Sigma ) ^ { 1 / 4 }$ and $B = \mathrm { T r } \Sigma / d$ , which gives the equivalent (7). Dividing the non-asymptotic bound by $\frac { 5 } { 8 } \sqrt { \mathrm { T r } \Sigma } \frac { \mathrm { l o } \mathrm { \ddot { g } } N } { N }$ , the ratio of the bound to the error is asymptotically

$$
a _ { N } \rho ^ { 1 / N } \frac { \log ( \rho ^ { 2 / 5 } N ) } { \log N } + \frac { 8 ( 1 + \sqrt { 8 } ) } { 5 \log N } = a _ { N } \rho ^ { 1 / N } \bigg ( 1 + \frac { 2 \log \rho } { 5 \log N } \bigg ) + \frac { 8 ( 1 + \sqrt { 8 } ) } { 5 \log N } \underset { N \to \infty } { \longrightarrow } 1
$$

for fixed $\Sigma ,$ since $a _ { N }  1 , \rho ^ { 1 / N }  1$ and $\rho$ does not depend on N. The bound is therefore asymptotically exact, which completes the proof of Theorem 3. □

## D Proof of Corollary 5 (ε-complexity)

This section proves the following corollary, stated in Section 2.3.

Corollary 5 (ε-complexity). Let $0 < \varepsilon \le \sqrt { \mathrm { T r } \Sigma }$ . In order for the sampling error to satisfy ${ \cal W } _ { 2 } ( { \mathbb P } _ { N } , { \mathbb P } ) \le \varepsilon ,$ it suffices

• for ULA with η = min ${ \left( { \lambda _ { \operatorname* { m i n } } , \varepsilon } / { \sqrt { \operatorname { T r } ( \Sigma ^ { - 1 } ) } } \right) }$ , to take

$$
N \geq \frac { \kappa } { 2 } \operatorname* { m a x } \left( 1 , \frac { \lambda _ { \operatorname* { m i n } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } { \varepsilon } \right) \log \frac { \sqrt { 8 \mathrm { T r } \Sigma } } { \varepsilon } ,
$$

$$
\bullet \ f o r \ U L D \ w i t h \ \eta = \operatorname* { m i n } \left( \sqrt { \lambda _ { \operatorname* { m i n } } / 8 } , \frac { 7 \sqrt { 2 } } { 6 } \varepsilon / \sqrt { \lambda _ { \operatorname* { m i n } } \mathrm { T r } ( \Sigma ^ { - 1 } ) } \right) a n d \gamma = \sqrt { 8 / \lambda _ { \operatorname* { m i n } } } , t o t a k e
$$

$$
N \geq 4 \kappa \operatorname* { m a x } \left( 1 , { \frac { 3 } { 1 4 } } { \frac { \lambda _ { \operatorname* { m i n } } { \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } } { \varepsilon } } \right) \log { \frac { 1 6 { \sqrt { \mathrm { T r } \Sigma } } } { \varepsilon } } ,
$$

• for the diffusion algorithm with $\delta = \varepsilon ^ { 2 } / ( 9 d ) , T = \sqrt { 3 / \varepsilon } \operatorname { T r } \left( ( \Sigma + \delta \operatorname { I } ) ^ { 3 } \Sigma ^ { 2 } \right) ^ { 1 / 4 }$ , and the geometric grid $\tau _ { k } = T ( \delta / T ) ^ { k / N }$ , to take

$$
N \geq \frac { 3 e } { 4 } \frac { \sqrt { \mathrm { T r } \Sigma + 4 \varepsilon ^ { 2 } / 9 } } { \varepsilon } \log \frac { T } { \delta } .
$$

The three ε-complexities follow from the parametric bounds of Propositions B.1, B.7 and C.1 respectively.

Proof. Let $\varepsilon \leq \sqrt { \mathrm { T r } \Sigma }$ . In each case we choose the hyperparameters and the budget N so that the error terms are all at most a fixed fraction of ε.

ULA. By Proposition B.1, for every $0 < \eta \leq \lambda _ { \mathrm { m i n } }$ and every $N \geq 0 .$

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \sqrt { 2 \operatorname { T r } \Sigma } e ^ { - 2 \eta N / \lambda _ { \operatorname* { m a x } } } + \frac { 1 } { 2 } \sqrt { \operatorname { T r } ( \Sigma ^ { - 1 } ) } \eta ,\tag{82}
$$

and we make each term at most $\varepsilon / 2$ . For the discretization term,

$$
\frac { 1 } { 2 } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \eta \le \frac { \varepsilon } { 2 } \iff \eta \le \frac { \varepsilon } { \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } .\tag{83}
$$

We take

$$
\eta = \operatorname* { m i n } { \left( \lambda _ { \operatorname* { m i n } } , \frac { \varepsilon } { \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } \right) } ,\tag{84}
$$

which is admissible for Proposition B.1 and saturates (83) whenever possible: it is the largest admissible step-size, hence the one that makes the exponential factor decay fastest. For the early-stopping term,

$$
\sqrt { 2 \mathrm { T r } \Sigma } e ^ { - 2 \eta N / \lambda _ { \mathrm { m a x } } } \leq \frac { \varepsilon } { 2 } \iff N \geq \frac { \lambda _ { \mathrm { m a x } } } { 2 \eta } \log \frac { \sqrt { 8 \mathrm { T r } \Sigma } } { \varepsilon } ,
$$

and with (84),

$$
\frac { \lambda _ { \operatorname* { m a x } } } { 2 \eta } = \frac { \lambda _ { \operatorname* { m a x } } } { 2 } \operatorname* { m a x } \left( \frac { 1 } { \lambda _ { \operatorname* { m i n } } } , \frac { \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } { \varepsilon } \right) = \frac { \kappa } { 2 } \operatorname* { m a x } \left( 1 , \frac { \lambda _ { \operatorname* { m i n } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } { \varepsilon } \right) .
$$

This is the announced condition

$$
N \geq \frac { \kappa } { 2 } \operatorname* { m a x } \left( 1 , \frac { \lambda _ { \operatorname* { m i n } } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } { \varepsilon } \right) \log \frac { \sqrt { 8 \mathrm { T r } \Sigma } } { \varepsilon } .
$$

ULD. By the first bound of Proposition B.7, for every $\gamma \geq \sqrt { 8 / \lambda _ { \operatorname* { m i n } } } .$ , every $\eta \le 1 / \gamma$ and every $N \geq 0$

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \le 8 \sqrt { \mathrm { T r } \Sigma } e ^ { - 2 \eta N / ( \lambda _ { \operatorname* { m a x } } \gamma ) } + \frac { 6 } { 7 } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \frac { \eta } { \gamma } .
$$

This bound depends on $( \eta , \gamma )$ only through $\phi : = \eta / \gamma$ , and the constraints translate into $\phi \leq 1 / \gamma ^ { 2 } \leq \lambda _ { \operatorname* { m i n } } / 8 ;$ conversely, every $\phi \in ( 0 , \lambda _ { \operatorname* { m i n } } / 8 ]$ is reached by an admissible pair, for instance $\gamma = \sqrt { 8 / \lambda _ { \operatorname* { m i n } } }$ and $\eta = \phi \gamma$ . We therefore work with the bound

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq 8 \sqrt { \mathrm { T r } \Sigma } e ^ { - 2 \phi N / \lambda _ { \operatorname* { m a x } } } + \frac { 6 } { 7 } \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } \phi , \quad \phi \leq \frac { \lambda _ { \operatorname* { m i n } } } { 8 } .
$$

Up to some numerical constants, this bound has the same form as in (82). We proceed as before and obtain the condition of Corollary 5, namely

$$
N \geq 4 \kappa \operatorname* { m a x } \left( 1 , { \frac { 3 } { 1 4 } } { \frac { \lambda _ { \operatorname* { m i n } } { \sqrt { \mathrm { T r } ( \Sigma ^ { - 1 } ) } } } { \varepsilon } } \right) \log { \frac { 1 6 { \sqrt { \mathrm { T r } \Sigma } } } { \varepsilon } } ,
$$

with

$$
\eta = \operatorname* { m i n } { \left( \sqrt { \frac { \lambda _ { \operatorname* { m i n } } } { 8 } } , \frac { 7 \sqrt { 2 } } { 6 } \frac { \varepsilon } { \sqrt { \lambda _ { \operatorname* { m i n } } \operatorname { T r } ( \Sigma ^ { - 1 } ) } } \right) } , \quad \gamma = \sqrt { \frac { 8 } { \lambda _ { \operatorname* { m i n } } } } .
$$

Diffusion. We show the more general statement of Remark 6, namely that, with

$$
\delta = \varepsilon ^ { 2 } / ( 9 d ) , \quad T = \sqrt { 3 / \varepsilon } \operatorname { T r } \left( ( \Sigma + \delta \operatorname { I } ) ^ { 3 } \Sigma ^ { 2 } \right) ^ { 1 / 4 }
$$

and the geometric grid $\tau _ { k } = T ( \delta / T ) ^ { k / N }$ , an error ε is achieved by taking, for any $r \geq 1$

$$
N \geq r \operatorname* { m a x } \left( 1 , \frac { 3 e ^ { 1 / r } } { 4 r } \frac { \sqrt { \mathrm { T r } \Sigma + 4 \varepsilon ^ { 2 } / 9 } } { \varepsilon } \right) \log \frac { T } { \delta } .
$$

By Proposition C.1, for every $N \geq 1$ and every $T > \delta > 0$

$$
W _ { 2 } ( \mathbb { P } _ { N } , \mathbb { P } ) \leq \frac { \sqrt { \mathrm { T r } \left( ( \Sigma + \delta { \mathrm { I } } ) ^ { 3 } \Sigma ^ { 2 } \right) } } { T ^ { 2 } } + \frac { \sqrt { \mathrm { T r } \Sigma + 4 d \delta } } { 4 N } \biggl ( \frac { T } { \delta } \biggr ) ^ { 1 / N } \log \frac { T } { \delta } + \sqrt { d \delta } ,
$$

and we make each of the three terms at most $\varepsilon / 3$

Early stopping. We have

$$
\sqrt { d \delta } \leq \frac { \varepsilon } { 3 } \Longleftrightarrow \delta \leq \frac { \varepsilon ^ { 2 } } { 9 d } .
$$

We saturate this constraint by taking

$$
\delta = \frac { \varepsilon ^ { 2 } } { 9 d } .\tag{85}
$$

Initialization. We have

$$
\frac { \mathrm { T r } \left( ( \Sigma + \delta \operatorname { I } ) ^ { 3 } \Sigma ^ { 2 } \right) ^ { 1 / 2 } } { T ^ { 2 } } \leq \frac { \varepsilon } { 3 } \Longleftrightarrow T \geq \sqrt { \frac { 3 } { \varepsilon } } \mathrm { T r } \left( ( \Sigma + \delta \operatorname { I } ) ^ { 3 } \Sigma ^ { 2 } \right) ^ { 1 / 4 } ,
$$

so we take

$$
T = \sqrt { \frac { 3 } { \varepsilon } } \mathrm { T r } \left( ( \Sigma + \delta \mathrm { I } ) ^ { 3 } \Sigma ^ { 2 } \right) ^ { 1 / 4 } .\tag{86}
$$

This choice is compatible with the constraint $T > \delta$ . Indeed, with (85) and (86),

$$
\frac { T } { \delta } = \left( \frac { 3 } { \varepsilon } \right) ^ { 5 / 2 } d \mathrm { T r } \left( ( \Sigma + \delta \mathrm { I } ) ^ { 3 } \Sigma ^ { 2 } \right) ^ { 1 / 4 } > \left( \frac { 3 } { \varepsilon } \right) ^ { 5 / 2 } d \mathrm { T r } ( \Sigma ^ { 5 } ) ^ { 1 / 4 } .\tag{87}
$$

But, by Holder’s inequality,¨ $\mathrm { T r } ( \Sigma ^ { 5 } ) \geq ( \mathrm { T r } \Sigma ) ^ { 5 } / d ^ { 4 }$ , so (87) yields

$$
\frac { T } { \delta } > 3 ^ { 5 / 2 } \left( \frac { \mathrm { T r } \Sigma } { \varepsilon ^ { 2 } } \right) ^ { 5 / 4 } > 1 ,\tag{88}
$$

since we assumed $\varepsilon \leq \sqrt { \mathrm { T r } \Sigma }$

Discretization. It remains to ensure

$$
E : = \frac { \sqrt { \mathrm { T r } \Sigma + 4 d \delta } } { 4 N } \Big ( \frac { T } { \delta } \Big ) ^ { 1 / N } \log \frac { T } { \delta } \leq \frac { \varepsilon } { 3 } .
$$

The factor $( T / \delta ) ^ { 1 / N }$ is what prevents solving $E \le \varepsilon / 3$ in closed form; but it tends to 1 as $N$ grows, and it is under control as soon as $N$ exceeds a few multiples of $\log ( T / \delta )$ . Precisely, for any $r \geq 1$

$$
\left( \frac { T } { \delta } \right) ^ { 1 / N } \leq e ^ { 1 / r } \Longleftrightarrow N \geq r \log \frac { T } { \delta } .\tag{89}
$$

Under (89) for a given $r \geq 1$

$$
E \le \frac { e ^ { 1 / r } } { 4 } \frac { \sqrt { \mathrm { T r } \Sigma + 4 d \delta } } { N } \log \frac { T } { \delta } ,
$$

and

$$
\frac { e ^ { 1 / r } } { 4 } \frac { \sqrt { \mathrm { T r } \Sigma + 4 d \delta } } { N } \log \frac { T } { \delta } \leq \frac { \varepsilon } { 3 } \Longleftrightarrow N \geq \frac { 3 e ^ { 1 / r } } { 4 } \frac { \sqrt { \mathrm { T r } \Sigma + 4 d \delta } } { \varepsilon } \log \frac { T } { \delta } .
$$

Both requirements are met as soon as

$$
N \geq \operatorname* { m a x } \left( r \log \frac { T } { \delta } , \frac { 3 e ^ { 1 / r } } { 4 } , \frac { \sqrt { \mathrm { T r } \Sigma + 4 d \delta } } { \varepsilon } \log \frac { T } { \delta } \right) = r \operatorname* { m a x } \left( 1 , \frac { 3 e ^ { 1 / r } } { 4 r } \frac { \sqrt { \mathrm { T r } \Sigma + 4 \varepsilon ^ { 2 } / 9 } } { \varepsilon } \right) \log \frac { T } { \delta } ,\tag{90}
$$

which is the budget of Remark 6. The parameter r tunes a trade-off: a larger r brings the constant $3 e ^ { 1 / r } / 4$ down towards $3 / 4$ , at the price of a higher floor $r \log ( T / \delta )$

Taking $r = 1$ , since $\textstyle { \sqrt { \operatorname { T r } \Sigma + 4 \varepsilon ^ { 2 } / 9 } } \geq { \frac { 2 } { 3 } } \varepsilon$ , the second argument of the maximum in (90) is at least $\textstyle { \frac { 3 e } { 4 } } \times { \frac { 2 } { 3 } } =$ $\frac { e } { 2 } > 1$ . The maximum is therefore always attained by the second argument, and (90) reduces to

$$
N \geq { \frac { 3 e } { 4 } } { \frac { \sqrt { \mathrm { T r } \Sigma + 4 \varepsilon ^ { 2 } / 9 } } { \varepsilon } } \log { \frac { T } { \delta } } ,\tag{91}
$$

the budget of Corollary 5.

Remark D.1 (Order of magnitude of $\log ( T / \delta ) ,$ ). We can make the logarithmic factor in (90) explicit. By (88), we have the lower bound

$$
\log { \frac { T } { \delta } } \geq { \frac { 5 } { 2 } } \log { \frac { 3 { \sqrt { \mathrm { T r } \Sigma } } } { \varepsilon } } .
$$

For the upper bound, with (85) we have

$$
\delta \leq \frac { \mathrm { T r } \Sigma } { 9 d } \leq \frac { 1 } { 9 } \lambda _ { \mathrm { m a x } } ,
$$

whence

$$
\mathrm { T r } ( ( \Sigma + \delta \operatorname { I } ) ^ { 3 } \Sigma ^ { 2 } ) \le ( \lambda _ { \operatorname* { m a x } } + \delta ) ^ { 3 } \lambda _ { \operatorname* { m a x } } \mathrm { T r } \Sigma \le \Big ( \frac { 1 0 } { 9 } \Big ) ^ { 3 } \lambda _ { \operatorname* { m a x } } ^ { 4 } \mathrm { T r } \Sigma .
$$

With $\rho = d \lambda _ { \operatorname* { m a x } } / \operatorname { T r } \Sigma \in [ 1 , d ]$ as in Theorem 3, (87) yields

$$
\frac { T } { \delta } \leq \Big ( \frac { 1 0 } { 9 } \Big ) ^ { 3 / 4 } \Big ( \frac { 3 } { \varepsilon } \Big ) ^ { 5 / 2 } d \lambda _ { \operatorname* { m a x } } ( \mathrm { T r } \Sigma ) ^ { 1 / 4 } = 3 ^ { 5 / 2 } \Big ( \frac { 1 0 } { 9 } \Big ) ^ { 3 / 4 } \rho \Big ( \frac { \sqrt { \mathrm { T r } \Sigma } } { \varepsilon } \Big ) ^ { 5 / 2 } .
$$

Since log $\left( 3 ^ { 5 / 2 } ( 1 0 / 9 ) ^ { 3 / 4 } \right) \approx 2 . 8 3 < 3 ,$ , we finally obtain

$$
{ \frac { 5 } { 2 } } \log { \frac { 3 { \sqrt { \operatorname { T r } \Sigma } } } { \varepsilon } } \leq \log { \frac { T } { \delta } } \leq { \frac { 5 } { 2 } } \log { \frac { \sqrt { \operatorname { T r } \Sigma } } { \varepsilon } } + \log \rho + 3 .\tag{92}
$$

In particular, $\log ( T / \delta ) = \Theta ( \log ( \sqrt { \mathrm { T r } \Sigma } / \varepsilon ) )$ for fixed Σ. Combined with $\textstyle \sqrt { \mathrm { T r } \Sigma + 4 \varepsilon ^ { 2 } / 9 } \leq \frac { \sqrt { 1 3 } } { 3 } \sqrt { \mathrm { T r } \Sigma } ,$ , equation (91) and the upper bound in (92) lead to the fully explicit budget condition

$$
N \geq { \frac { { \sqrt { 1 3 } } e } { 4 } } { \frac { \sqrt { \operatorname { T r } \Sigma } } { \varepsilon } } { \left( { \frac { 5 } { 2 } } \log { \frac { \sqrt { \operatorname { T r } \Sigma } } { \varepsilon } } + \log \rho + 3 \right) } ,
$$

of order $\tilde { O } ( \sqrt { \mathrm { T r } \Sigma } / \varepsilon )$ with no dependence on κ.

## E Proof of Theorem 7 (score learning)

This section proves the following theorem, stated in Section 3.

Theorem 7 (Score learning). Consider gradient descent initialized at $A _ { 0 } = 0 _ { i }$ , with step-size $\alpha > 0$ , on both L and $L _ { \sigma }$ . Let $K \geq 1$

$\begin{array} { r } { I f \alpha < \frac { 2 } { \hat { \lambda } _ { \mathrm { m a x } } } } \end{array}$ , the iterate $A _ { K }$ is invertible and

$$
A _ { K } ^ { - 1 } = \hat { \Sigma } + \frac { 1 } { K \alpha } \Pi _ { \mathrm { K e r } \hat { \Sigma } } + O ( \rho _ { 1 } ^ { K } ) ,
$$

with $\rho _ { 1 } = \operatorname* { m a x } \big ( | 1 - \alpha \widehat { \lambda } _ { \operatorname* { m i n } } ^ { + } | , | 1 - \alpha \widehat { \lambda } _ { \operatorname* { m a x } } | \big ) < 1$

$\begin{array} { r } { I f \alpha < \frac { 2 } { \hat { \lambda } _ { \mathrm { m a x } } + \sigma ^ { 2 } } } \end{array}$ , the iterate $A _ { K } ^ { \sigma }$ is invertible and

$$
( A _ { K } ^ { \sigma } ) ^ { - 1 } = \hat { \Sigma } + \sigma ^ { 2 } { \mathrm { I } } + O ( \rho _ { 2 } ^ { K } ) ,
$$

$$
w i t h \rho _ { 2 } = \operatorname* { m a x } \big ( | 1 - \alpha ( \hat { \lambda } _ { \operatorname* { m i n } } + \sigma ^ { 2 } ) | , | 1 - \alpha ( \hat { \lambda } _ { \operatorname* { m a x } } + \sigma ^ { 2 } ) | \big ) < 1 .
$$

The proof is a direct computation: we first write both losses in closed form and derive their gradients, then compute the iterates $A _ { K }$ and $A _ { K } ^ { \sigma }$ explicitly, and finally invert these expressions.

## E.1 Losses and gradients

We treat the two losses simultaneously by introducing, for $t \geq 0$

$$
L ^ { ( t ) } ( A ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathbb { E } _ { \omega } [ \nabla \cdot s _ { A } ( x _ { i } + \omega ) + \frac { 1 } { 2 } \| s _ { A } ( x _ { i } + \omega ) \| ^ { 2 } ] ,
$$

with $\omega \sim \mathcal { N } ( 0 , t \mathrm { I } )$ . Hence, $L = L ^ { ( 0 ) }$ and $L _ { \sigma } = L ^ { ( \sigma ^ { 2 } ) }$ . We first compute the losses and their gradients:

Lemma E.1. For all $t \geq 0$ and $A \in \mathbb { R } ^ { d \times d }$ , we have

$$
L ^ { ( t ) } ( A ) = \frac { 1 } { 2 } \operatorname { T r } ( A \hat { \Sigma } A ^ { \top } ) - \operatorname { T r } A + \frac { t } { 2 } \| A \| _ { F } ^ { 2 } ,
$$

with gradient

$$
\nabla L ^ { ( t ) } ( A ) = A ( \hat { \Sigma } + t { \mathrm { I } } ) - \boldsymbol { \mathrm { I } } .
$$

Proof. Fix $t \geq 0 , A \in \mathbb { R } ^ { d \times d }$ . For any $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , we have $s _ { A } ( x ) = - A x .$ , hence

$$
\begin{array} { l } { \displaystyle { I ^ { ( t ) } ( A ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \mathbb { E } _ { \omega } \big [ - \mathsf { T r } A + \frac { 1 } { 2 } \| A ( x _ { i } + \omega ) \| ^ { 2 } \big ] } } \\ { \displaystyle { \quad = - \mathsf { T r } A + \frac { 1 } { 2 m } \sum _ { i = 1 } ^ { m } \mathbb { E } _ { \omega } \big [ \| A x _ { i } \| ^ { 2 } + 2 x _ { i } ^ { \top } A ^ { \top } A \omega + \| A \omega \| ^ { 2 } \big ] } } \\ { \displaystyle { \quad = - \mathrm { T r } A + \frac { 1 } { 2 } \mathbb { E } _ { \omega } \big [ \varpi ^ { \top } A ^ { \top } A \omega \big ] + \frac { 1 } { 2 m } \sum _ { i = 1 } ^ { m } \mathrm { T r } \big ( A x _ { i } x _ { i } ^ { \top } A ^ { \top } \big ) + \frac { 1 } { m } \Bigg ( \sum _ { i = 1 } ^ { m } x _ { i } ^ { \top } A ^ { \top } A \Big ) \mathbb { E } \omega } } \\ { \displaystyle { \quad = - \mathrm { T r } A + \frac { 1 } { 2 } \mathrm { T r } ( t \mathrm { I A } A ^ { \top } ) + \frac { 1 } { 2 } \mathrm { T r } ( A \hat { \Sigma } A ^ { \top } ) } } \\ { \displaystyle { \quad = \frac { 1 } { 2 } \mathrm { T r } ( A \hat { \Sigma } A ^ { \top } ) - \mathrm { T r } A + \frac { t } { 2 } \| A \| _ { F } ^ { 2 } . } } \end{array}
$$

Denoting

$$
f : A \longmapsto { \frac { 1 } { 2 } } \operatorname { T r } ( A { \hat { \Sigma } } A ^ { \top } ) , \quad g : A \longmapsto - \operatorname { T r } A , \quad h : A \longmapsto { \frac { 1 } { 2 } } \| A \| _ { F } ^ { 2 } ,
$$

we compute, using the symmetry of $\hat { \Sigma } .$

$$
\nabla f ( A ) = A \hat { \Sigma } , \quad \nabla g ( A ) = - \operatorname { I } , \quad \nabla h ( A ) = A .
$$

The gradient of $L ^ { ( t ) }$ then follows:

$$
\nabla L ^ { ( t ) } ( A ) = A ( \hat { \Sigma } + t { \mathrm { I } } ) - \boldsymbol { \mathrm { I } } .
$$

Note that in particular, we retrieve the fact that

$$
\forall A , \quad L _ { \sigma } ( A ) = L ( A ) + { \frac { \sigma ^ { 2 } } { 2 } } \| A \| _ { F } ^ { 2 } ,
$$

so the loss $L _ { \sigma }$ is indeed a ridge-type regularization of $L ,$ as claimed in Section 3.

## E.2 Closed form of the iterates

Let $\hat { \Sigma } _ { t } = \hat { \Sigma } + t \mathrm { I }$ and denote by $( \hat { \lambda } _ { i } ) _ { 1 \leq i \leq d }$ the eigenvalues of ${ \hat { \Sigma } } .$ with an orthonormal eigenbasis $( u _ { i } ) ; \hat { \Sigma } _ { t }$ has eigenvalues $\hat { \lambda } _ { i } + t$ in the same basis. Moreover, denote by $\hat { \Sigma } ^ { + }$ the Moore–Penrose inverse of $\hat { \Sigma } , i . e .$ , the matrix defined by

$$
\hat { \Sigma } ^ { + } = \sum _ { \scriptstyle i = 1 \atop \hat { \lambda } _ { i } \neq 0 } ^ { d } \frac { 1 } { \hat { \lambda } _ { i } } u _ { i } u _ { i } ^ { \top } .
$$

Proposition E.2. Let $\alpha > 0$ be the gradient step-size. For every $K \geq 0 ,$ , the iterate of gradient descent on $L ^ { ( t ) }$ is

$$
A _ { K } ^ { ( t ) } = \alpha \sum _ { j = 0 } ^ { K - 1 } ( \mathrm { I } - \alpha \hat { \Sigma } _ { t } ) ^ { j } = \sum _ { i = 1 } ^ { d } \psi _ { K } ( \hat { \lambda } _ { i } + t ) u _ { i } u _ { i } ^ { \top } ,
$$

with

$$
\psi _ { K } : x \longmapsto { \left\{ \begin{array} { l l } { { \frac { 1 - ( 1 - \alpha x ) ^ { K } } { x } } } & { i f x \neq 0 , } \\ { K \alpha } & { o t h e r w i s e . } \end{array} \right. }
$$

Hence, for $t > 0$ we have

$$
A _ { K } ^ { ( t ) } = \hat { \Sigma } _ { t } ^ { - 1 } \big ( \mathrm { I } - ( \mathrm { I } - \alpha \hat { \Sigma } _ { t } ) ^ { K } \big ) ,\tag{93}
$$

while

$$
A _ { K } ^ { ( 0 ) } = \hat { \Sigma } ^ { + } \big ( \mathrm { I } - ( \mathrm { I } - \alpha \hat { \Sigma } ) ^ { K } \big ) + K \alpha \Pi _ { \mathrm { K e r } \hat { \Sigma } } ,\tag{94}
$$

with $\hat { \Sigma } ^ { + } = \hat { \Sigma } ^ { - 1 }$ and $\Pi _ { \mathrm { K e r } \hat { \Sigma } } = 0$ when $\hat { \Sigma }$ is invertible.

Proof. Let $K \geq 0$ . The process $( A _ { j } ^ { ( t ) } ) _ { 0 \leq j \leq K }$ is defined by the gradient descent on the loss $L ^ { ( t ) }$ with step-size $\alpha .$ initialized at $A _ { 0 } ^ { ( t ) } = 0 \colon$

$$
\left\{ \begin{array} { l l } { \forall j , \quad A _ { j + 1 } ^ { ( t ) } = A _ { j } ^ { ( t ) } - \alpha \nabla L ^ { ( t ) } ( A _ { j } ^ { ( t ) } ) , } \\ { A _ { 0 } ^ { ( t ) } = 0 . } \end{array} \right.
$$

Therefore, by Lemma E.1, we have

$$
\forall j , \quad A _ { j + 1 } ^ { ( t ) } = A _ { j } ^ { ( t ) } ( \mathrm { I } - \alpha \hat { \Sigma } _ { t } ) + \alpha \mathrm { I } ,
$$

thus

$$
\begin{array} { l } { { \displaystyle { \cal A } _ { K } ^ { ( t ) } = \alpha \sum _ { j = 0 } ^ { K - 1 } ( \mathrm { I } - \alpha \hat { \Sigma } _ { t } ) ^ { j } } } \\ { { \displaystyle ~ = \sum _ { i = 1 } ^ { d } \bigg ( \alpha \sum _ { j = 0 } ^ { K - 1 } \big ( 1 - \alpha \big ( \hat { \lambda } _ { i } + t \big ) \big ) ^ { j } u _ { i } u _ { i } ^ { \top } \bigg ) } . } \end{array}
$$

The equality

$$
A _ { K } ^ { ( t ) } = \sum _ { i = 1 } ^ { d } \psi _ { K } ( \hat { \lambda } _ { i } + t ) u _ { i } u _ { i } ^ { \top }
$$

is then a simple consequence of the identity

$$
\sum _ { j = 0 } ^ { K - 1 } x ^ { j } = { \left\{ \begin{array} { l l } { { \frac { 1 - x ^ { K } } { 1 - x } } } & { { \mathrm { i f ~ } } x \neq 1 , } \\ { K } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }
$$

and the relations (93) and (94) follow.

## E.3 Inverses of the iterates

We can now prove Theorem 7.

Proof of Theorem 7. Fix $K \geq 1$ . By Proposition E.2, we have for any $t \geq 0 , \alpha > 0 ;$

$$
A _ { K } ^ { ( t ) } = \sum _ { i = 1 } ^ { d } \psi _ { K } ( \hat { \lambda } _ { i } + t ) u _ { i } u _ { i } ^ { \top } .
$$

In particular, for every $\begin{array} { r } { \alpha < \frac { 2 } { \hat { \lambda } _ { \mathrm { { m a x } } } + t } } \end{array}$ , the eigenvalues $\psi _ { K } ( \hat { \lambda } _ { i } + t )$ of $A _ { K } ^ { ( t ) }$ are all positive, thus non-zero, and $A _ { K } ^ { ( t ) }$ is invertible with

$$
( A _ { K } ^ { ( t ) } ) ^ { - 1 } = \sum _ { i = 1 } ^ { d } \frac { u _ { i } u _ { i } ^ { \top } } { \psi _ { K } ( \hat { \lambda } _ { i } + t ) } .\tag{95}
$$

Loss $L \left( t = 0 \right)$ . We have

$$
\psi _ { K } ( \hat { \lambda } _ { i } ) = \left\{ \begin{array} { l l } { \frac { 1 - ( 1 - \alpha \hat { \lambda } _ { i } ) ^ { K } } { \hat { \lambda } _ { i } } } & { \mathrm { i f } \ \hat { \lambda } _ { i } \neq 0 , } \\ { K \alpha } & { \mathrm { o t h e r w i s e , } } \end{array} \right.
$$

equation (95) therefore yields for $t = 0$

$$
\begin{array} { r l } {  { A _ { K } ^ { - 1 } - \sum _ { k = 1 } ^ { d } \frac { u _ { k } u _ { k } ^ { \top } } { \psi _ { K } ( \hat { \lambda } _ { k } ) } + \sum _ { k = 1 } ^ { d } \frac { u _ { k } u _ { k } ^ { \top } } { \psi _ { K } ( \hat { \lambda } _ { k } ) } } } \\ & { = \sum _ { k = 1 } ^ { d } \frac { \hat { \lambda } _ { k } } { \psi _ { K } ( \hat { \lambda } _ { k } ) } - \sum _ { k = 0 } ^ { d } \frac { \hat { \lambda } _ { k } } { \psi _ { K } ( \hat { \lambda } _ { k } ) } u _ { k } ^ { - 1 } \sum _ { k = 1 } ^ { d } \frac { 1 } { K \alpha } u _ { i k } ^ { - 1 } } \\ & { = \sum _ { k = 0 } ^ { d } \frac { 1 } { \hat { \lambda } _ { k } \psi _ { 0 } } } \\ & { = \sum _ { k = 0 } ^ { d } \hat { \lambda } _ { k } u _ { k } u _ { k } ^ { \top } + \sum _ { k = 1 } ^ { d } \frac { \hat { \lambda } _ { k } ( 1 - \alpha \hat { \lambda } _ { i } ) ^ { K } } { \hat { \lambda } _ { k } \psi _ { 0 } } u _ { k } ^ { \top } + \frac { 1 } { K \alpha } \Pi _ { K \kappa \hat { \lambda } } \frac { 1 } { K \alpha } \Pi _ { K \kappa \hat { \lambda } } ^ { 2 } } \\ & { = \sum _ { k = 0 } ^ { d - 1 } \frac { 1 } { K \alpha } u _ { k } ^ { \top } \hat { \lambda } _ { k } - ( 1 - \alpha \hat { \lambda } _ { i } ) ^ { K } u _ { k } ^ { \top } u _ { i } ^ { - 1 } + \frac { 1 } { K \alpha } \Pi _ { K \kappa \hat { \lambda } } \frac { 1 } { K \alpha } } \\ &  = \hat { \lambda } + \frac { 1 } { K \alpha } \Pi _ { K \alpha \hat { \lambda } } + \sum _ { k = 0 } ^ { d } \frac  \hat { \lambda } \end{array}\tag{96}
$$

Denote by $R _ { K }$ the last sum. It is a symmetric matrix, diagonal in the basis $( u _ { i } )$ , whose eigenvalues are the numbers

$$
r _ { i } : = \frac { \hat { \lambda } _ { i } ( 1 - \alpha \hat { \lambda } _ { i } ) ^ { K } } { 1 - ( 1 - \alpha \hat { \lambda } _ { i } ) ^ { K } } , \quad \hat { \lambda } _ { i } \neq 0 .
$$

Since $x \mapsto \left| 1 - \alpha x \right|$ is convex, its maximum over the interval $[ \hat { \lambda } _ { \operatorname* { m i n } } ^ { + } , \hat { \lambda } _ { \operatorname* { m a x } } ]$ is attained at one of the endpoints: $| 1 - \alpha \hat { \lambda } _ { i } | \leq$ max $\big ( | 1 - \alpha \hat { \lambda } _ { \mathrm { m i n } } ^ { + } | , | 1 - \alpha \hat { \lambda } _ { \mathrm { m a x } } | \big ) = \rho _ { 1 }$ for every i with $\hat { \lambda } _ { i } \neq 0$ . The assumption $\alpha < 2 / \hat { \lambda } _ { \mathrm { m a x } }$ gives $\rho _ { 1 } < 1$ ; therefore

$$
| r _ { i } | \leq \frac { \hat { \lambda } _ { i } \rho _ { 1 } ^ { K } } { 1 - \rho _ { 1 } ^ { K } } \leq \frac { \hat { \lambda } _ { i } } { 1 - \rho _ { 1 } } \rho _ { 1 } ^ { K } .
$$

Hence, denoting $\lVert \cdot \rVert _ { 2 }$ the spectral norm, we have

$$
\| R _ { K } \| _ { 2 } \leq \frac { \hat { \lambda } _ { \operatorname* { m a x } } } { 1 - \rho _ { 1 } } \rho _ { 1 } ^ { K } = O ( \rho _ { 1 } ^ { K } ) ,
$$

which, along with (96), leads to the first claim of Theorem 7.

Loss $L _ { \sigma } \left( t = \sigma ^ { 2 } \right)$ . The argument is the same, without any kernel to single out; the inverse formula (95) gives

$$
\begin{array} { l } { ( A _ { K } ^ { \sigma } ) ^ { - 1 } = \displaystyle \sum _ { i = 1 } ^ { d } \frac { \hat { { \lambda } } _ { i } + \sigma ^ { 2 } } { 1 - ( 1 - \alpha ( \hat { { \lambda } } _ { i } + \sigma ^ { 2 } ) ) ^ { K } } u _ { i } u _ { i } ^ { \top } } \\ { = \displaystyle \sum _ { i = 1 } ^ { d } \bigg ( \hat { { \lambda } } _ { i } + \sigma ^ { 2 } + \frac { ( \hat { { \lambda } } _ { i } + \sigma ^ { 2 } ) ( 1 - \alpha ( \hat { { \lambda } } _ { i } + \sigma ^ { 2 } ) ) ^ { K } } { 1 - ( 1 - \alpha ( \hat { { \lambda } } _ { i } + \sigma ^ { 2 } ) ) ^ { K } } \bigg ) u _ { i } u _ { i } ^ { \top } } \\ { = \hat { \Sigma } + \sigma ^ { 2 } \operatorname { I } + R _ { K } ^ { \sigma } , } \end{array}\tag{97}
$$

where

$$
R _ { K } ^ { \sigma } = \sum _ { i = 1 } ^ { d } \frac { ( \hat { \lambda } _ { i } + \sigma ^ { 2 } ) ( 1 - \alpha ( \hat { \lambda } _ { i } + \sigma ^ { 2 } ) ) ^ { K } } { 1 - ( 1 - \alpha ( \hat { \lambda } _ { i } + \sigma ^ { 2 } ) ) ^ { K } } u _ { i } u _ { i } ^ { \top } .
$$

Under the assumption $\alpha < 2 / ( \hat { \lambda } _ { \mathrm { m a x } } + \sigma ^ { 2 } )$

$$
\| R _ { K } ^ { \sigma } \| _ { 2 } \leq \frac { \hat { \lambda } _ { \operatorname* { m a x } } + \sigma ^ { 2 } } { 1 - \rho _ { 2 } } \rho _ { 2 } ^ { K } = O ( \rho _ { 2 } ^ { K } ) ,
$$

with $\rho _ { 2 } = \operatorname* { m a x } \big ( | 1 - \alpha ( \hat { \lambda } _ { \operatorname* { m i n } } + \sigma ^ { 2 } ) | , | 1 - \alpha ( \hat { \lambda } _ { \operatorname* { m a x } } + \sigma ^ { 2 } ) | \big )$ . With (97), this gives the second claim of Theorem 7.

## LLM usage

The appendix, initially written by the authors in another language, was translated into English with the help of LLMs. LLMs were also used to assist in checking the proofs and in writing the code that generates the figures. All LLM-assisted content was reviewed and verified by the authors.

## Code availability

The code used to generate the figures is available at

$$
\mathrm { { h t t p s : / / g i t h u b . c o m / a p e r b o s t / d i f f u s i o n \mathrm { { \overline { { { \Omega } } } \mathrm { { s . } } \mathrm { { - } \mathrm { { } { L a n g e v i n . } } } } } } }
$$

The Wasserstein distances are computed in closed form from the expressions of the eigenvalues of the covariance matrices given in the Appendices B.2.1, B.3.1 and C.2.1, so no stochastic simulation is involved.