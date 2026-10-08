# Pre-training of Bayesian Optimization Algorithm through Bayesian Optimization

Satoshi Katayama<sup>1</sup>, Shoyo Hunt<sup>1</sup>, Shintaro Masuda<sup>1</sup>, and Masayuki Karasuyama<sup>∗1</sup>

<sup>1</sup>Nagoya Institute of Technology

## Abstract

Bayesian optimization (BO) is widely used as a standard approach for expensive black box optimization. However, BO algorithms often involve parameters that must be specified in advance, and their performance can strongly depend on these choices. We propose a framework for optimizing such parameters using sample paths drawn from a Gaussian process (GP) inferred from the information available at the start of BO. We use cumulative regret as the performance metric for a BO algorithm. By running the BO algorithm on the generated sample paths, we obtain an empirical estimate of its expected cumulative regret for a given parameter configuration. Optimizing this estimate allows us to identify parameter configurations that, given the currently available information, are expected to achieve low cumulative regret. Since this parameter optimization is itself a black-box optimization problem, we employ another BO procedure to solve it, which we refer to as outer BO. Through experiments, we demonstrate that the proposed framework can efectively select parameter configurations that achieve strong performance among a range of candidate configurations.

## 1 Introduction

Bayesian optimization (BO) is a widely used approach for optimizing expensive black-box functions with a limited number of function evaluations. It has been successfully applied to a broad range of problems, including experimental design, hyperparameter optimization, and scientific and engineering design. However, while BO automates the optimization of the objective function, BO algorithms themselves often involve various algorithmic configuration parameters that must be specified before optimization begins. Examples include parameters controlling the exploration–exploitation trade-of in acquisition functions and design parameters of search strategies. Such choices can sub stantially afect optimization performance. Unlike the surrogate model, i.e., usually a Gaussian process (GP), hyperparameters, these algorithmic parameters cannot generally be estimated by the (marginal) likelihood of the surrogate and are therefore often set to default or heuristic values.

The basic question in this study is whether the same criterion used to evaluate the performance of a BO algorithm can also be used to select its configuration parameters. We consider cumulative regret over a finite horizon as the performance criterion. The cumulative regret with respect to the unknown true objective function is, of course, unavailable before running BO. We assume that, at the beginning of BO, we have a probability distribution over the objective function that reflects the available initial observations and/or prior knowledge. We use the expected cumulative regret under this distribution, i.e., the Bayes cumulative regret, as the criterion. By sampling objective functions from this distribution and actually running a candidate BO algorithm on each sample path, we can obtain a Monte Carlo (MC) approximation of the expected cumulative regret. This allows us to formulate the selection of BO algorithm parameters as an optimization problem that directly targets finite-horizon performance under the currently available knowledge about the objective function.

An important feature of this approach is that prior knowledge about the objective function is used not only for prediction by the surrogate model but also for configuring the BO algorithm itself. In the simplest setting, the GP posterior conditioned on an initial dataset can be used as the distribution over objective functions. When data from related past tasks are available, a GP prior pre-trained on such data can instead be incorporated. Our framework therefore does not rely on a particular method for constructing the prior distribution; rather, it can be viewed as a general mechanism for translating an available distribution over objective functions into a configuration of a BO algorithm. Recent work on learning to optimize and meta-learning has proposed methods that learn optimizers or acquisition policies themselves from multiple optimization tasks (e.g., Chen et al., 2022; Volpp et al., 2020). In contrast, our approach retains the structure of an existing BO algorithm and tunes only a small number of its algorithmic parameters. This design aims to adapt the algorithm to the target problem distribution while retaining established acquisition rules and principles developed in the BO literature. We discuss these connections in more detail in Section 5.

The MC approximation of the cumulative regret is generally a complicated black-box function of the parameter configuration, for which an analytical expression or gradient is dificult to obtain. Moreover, evaluating a single configuration requires repeatedly running finite-horizon BO on multiple sample paths, making exhaustive evaluation impractical. We therefore solve this algorithm configuration problem using another BO procedure. We refer to the conventional BO procedure for optimizing the objective function as inner BO, and to the BO procedure for optimizing its al gorithmic parameters as outer BO. This construction makes the framework applicable to arbitrary parameterized BO algorithms without requiring an analytical characterization of either their internal mechanisms or the dependence of regret on their parameters. We further extend the framework to a multi-fidelity setting, where evaluations obtained with shorter inner-BO horizons are used as lower-fidelity observations, and to a multi-stage configuration setting, where diferent parameter values can be used at diferent stages of the optimization.

We also analyze the statistical properties of the proposed MC-based evaluation of BO algorithms. We show that the MC estimator of cumulative regret has approximation error $O _ { p } ( T / { \sqrt { N } } )$ , where T is the number of inner-BO iterations and N is the number of MC samples. When the candidate parameter configurations form a finite set, we further establish a guarantee of the same order for the diference in expected cumulative regret between the configuration selected by minimizing the MC approximation and the configuration minimizing the true Bayes cumulative regret. These results do not depend on the specific form of the acquisition function and provide theoretical justification for our simulation-based evaluation framework.

Our main contributions are summarized as follows:

• We propose a general framework for selecting pre-specified parameters of a BO algorithm using its expected cumulative regret under a probability distribution over objective functions. We approximate this criterion by running BO on sample paths and use outer BO to optimize the resulting black-box objective. We further extend the framework to multi-fidelity optimization using the inner-BO horizon as a fidelity variable and to multi-stage configurations in which algorithmic parameters can change during optimization.

• We establish the $O _ { p } ( T / { \sqrt { N } } )$ convergence of the MC cumulative-regret estimator. For a finite candidate configurations, we further provide a guarantee on the expected cumulative regret of the configuration selected using the MC estimator.

• Through experiments, we demonstrate that the proposed framework can identify parameter configurations with strong finite-horizon BO performance among a set of candidate configurations.

## 2 Bayesian Optimization (BO)

## 2.1 Preliminary

We consider the maximization problem of an unknown function $f : \mathcal { X } \to \mathbb { R }$ , whose input domain is d-dimensional compact space $\mathcal { X } \subset \mathbb { R } ^ { d }$ , i.e., $\operatorname* { m a x } _ { { \pmb x } \in { \pmb X } } f ( { \pmb x } )$ . An observation for $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ is denoted as $y _ { i } = f ( \pmb { x } _ { i } ) + \varepsilon$ , where ε is an observation noise that follows $\varepsilon \sim \mathcal N \left( 0 , \sigma _ { \mathrm { n o i s e } } ^ { 2 } \right)$

In Bayesian optimization (BO), the Gaussian process (GP) is typically used to model the unknown objective function $f ,$ denoted as

$$
f \sim { \mathcal { G P } } ( m , k ) ,
$$

where m is a prior mean function and k is a prior covariance function (kernel function), both of which are assumed to be continuous. Let $\mathcal { D } _ { t }$ be the dataset at the t-th iteration of BO. The posterior of f is

$$
f \mid { \mathcal { D } } _ { t } \sim { \mathcal { G P } } ( \mu _ { t } , \Sigma _ { t } ) ,
$$

where $\mu _ { t } ( \pmb { x } ) = \mathbb { E } [ f ( \pmb { x } ) \ | \ \mathcal { D } _ { t } ]$ and $\Sigma _ { t } ( \pmb { x } , \pmb { x } ^ { \prime } ) = \operatorname { C o v } ( f ( \pmb { x } ) , f ( \pmb { x } ^ { \prime } ) \ | \ \mathcal { D } _ { t } )$ are the posterior mean and covariance functions, respectively. By maximizing an acquisition function $^ { a , }$ the next observation point is determined as

$$
\operatorname* { m a x } _ { { \pmb x } \in { \mathscr X } } a ( { \pmb x } ) .
$$

Well known acquisition functions are for example EI (Expected Improvement) and GP-UCB (Gaussian process Upper Confidence Bound) (Srinivas et al., 2010). Throughout the paper, we assume that the number of the maximum iterations of BO is $T ,$

## 2.2 Parameter Configuration of BO

For a variety of problem settings, a lot of BO algorithms have been proposed (Wang et al., 2023). However, many of BO algorithms need configurations of parameters that are required to determine before starting the optimization. Let $\pmb { \theta } \in \Theta \subset \mathbb { R } ^ { p }$ be those parameters of a BO algorithm. Examples of θ are as follows.

EI. EI is often defined as the expected improvement compared with the maximum value among the observed $y _ { i }$ in the current dataset. However, this threshold actually can be set arbitrary (Brochu et al., 2010), which obviously has substantial efect on the results.

TuRBO. TuRBO (Trust Region BO) proposed by (Eriksson et al., 2019) is a well-known scalable BO method that performs eficient global optimization by adaptively managing multiple local trust regions (TRs), each modeled by an independent GPs. However, for example, the sizes of TRs and their change scheduling can be tuning parameters that afect the final performance.

```latex
Algorithm 1: BO algorithm
1 function $\operatorname { B O } ( f , \pmb \theta , T )$
2 for $i = 0 , \ldots , T$ do
3 Fit surrogate model (GP) to $\mathcal { D } _ { i }$
4 $\pmb { x } _ { t } \gets \mathrm { a r g m a x } _ { \pmb { x } \in \mathcal { X } } a ( \pmb { x } )$
5 Query observation $y _ { t } \gets f ( \pmb { x } _ { t } ) + \varepsilon$
6 $\mathcal { D } _ { t + 1 }  \mathcal { D } _ { t } \cup \{ ( x _ { t } , y _ { t } ) \}$
7 end
8 end
```

LSBO. LSBO (Latent Space BO) (G´omez-Bombarelli et al., 2018) performs BO in a lowdimensional latent space learned by a generative model, enabling eficient optimization over complex structured objects (such as molecules) or a higher dimensional original space. For the latent space embedding, variational auto-encoder (VAE) (Kingma and Welling, 2014) is typically used. However, the estimation in a VAE involves several design choices, such as balancing parameters of the reconstruction loss and the regularizations (such as a Gaussianity and independence in the latent space), and the dimensionality of the latent space.

Note that these parameters are configurations of a BO algorithm itself, and are not parameters of GPs, by which the standard hyper-parameter tuning by the marginal likelihood is not applicable. Because of this dificulty, these configurations are often set heuristically. However, an appropriate way of a configuration selection is obviously important for maximally accelerating the optimization. Algorithm 1 shows the process of BO algorithm. In the scenarios described above, typically, line 3-4 of the algorithm (surrogate fitting and acquisition function maximization) depends on θ.

## 3 Proposed Method: Pre-trained BO

## 3.1 Evaluation of BO Algorithm

The basic idea of our approach is that the evaluation criterion of an entire BO algorithm should be optimized to select an appropriate θ. For this purpose, we employ the cumulative regret, which is often used in empirical evaluation of BO algorithms,

$$
R _ { T } : = \sum _ { t = 1 } ^ { T } ( f ( \pmb { x } ^ { \star } ) - f ( \hat { \pmb { x } } _ { t } ) ) ,
$$

where $\pmb { x } ^ { \star } = \mathrm { a r g m a x } _ { \pmb { x } \in \pmb { \chi } } f ( \pmb { x } )$ and $\hat { \mathbf { x } } _ { t }$ is the recommended point at the t-th iteration. For evaluation of BO algorithms, typically, $\hat { \mathbf { x } } _ { t }$ is defined as the maximizer of the acquisition function, or the maximizer of the posterior mean is also often used (called inference regret).

Our proposed method, called pre-trained $B 0 ,$ considers minimizing $R _ { T }$ with respect to $\pmb { \theta } .$ However, for this purpose, we need to evaluate $R _ { T }$ before starting the actual BO procedure. Therefore, we use the expected cumulative regret (also called Bayes regret) measured by the initial posterior

$f \mid \mathcal { D } _ { 0 } \sim \mathcal { G P } ( \mu _ { 0 } , \Sigma _ { 0 } )$ , written as

$$
\mathrm { B R } _ { T } ( \pmb \theta ) : = \mathbb { E } [ R _ { T } \ | \ \mathcal D _ { 0 } ] .
$$

The expectation here is about $f , \varepsilon ,$ and other randomness contained in the BO algorithm (if exists). Since $\mu _ { 0 }$ and $\Sigma _ { 0 }$ are estimated by the initial dataset $\mathcal { D } _ { 0 }$ , they are available before starting the BO loop. The following optimization problem characterizes the best θ in a sense of the expected cumulative regret defined by the current knowledge of the optimization task.

$$
\operatorname* { m i n } _ { \pmb { \theta } \in \Theta } \ \mathrm { B R } _ { T } ( \pmb { \theta } )\tag{1}
$$

The posterior $\mu _ { 0 }$ and $\Sigma _ { 0 }$ can be purely estimated from $\mathcal { D } _ { 0 }$ , but richer prior knowledge can also be incorporated through the prior mean m and covariance k. We consider the following two typical scenarios.

• The first scenario is that m are manually created from some expert knowledge. In some application problems, human experts sometimes have background knowledge about $f .$ . For example, when $f$ is an expensive black-box function determined by quantum mechanics cal culation, simpler explicit function from classical mechanics can be used as m (e.g., Deringer et al., 2021).

• The second scenario is when prior (past) datasets are available. Pre-trained GP (Wang et al., 2024) is proposed for this setting, estimating m and k under the assumption that the past datasets share a common prior. For example, Wang et al. (2024) demonstrate this setting in AutoML, where historical hyperparameter configurations and their performance on related tasks are used to learn a prior for a new task. Another plausible setting is scientific experimen tation, where conditions and outcomes from related past experiments could similarly provide a prior for optimizing a new experiment.

## 3.2 Monte-Calro Approximation and Optimization

Since the expectation in $\mathrm { B R } _ { T }$ is computationally intractable, we employ the Monte-Carlo (MC) approximation. Let $f ^ { ( 1 ) } , \ldots , { \bar { f } } ^ { ( N ) }$ be N sample paths generated from $\mathcal { G P } ( \mu _ { 0 } , \Sigma _ { 0 } )$ . By performing BO T iterations for each of $f ^ { ( i ) }$ , we can obtain a realization of $R _ { T }$ , denoted as $R _ { T } ^ { ( i ) } \ =$ $\begin{array} { r } { \sum _ { t = 1 } ^ { T } ( f ^ { ( i ) } ( \pmb { x } _ { ( i ) } ^ { \star } ) - f ^ { ( i ) } ( \hat { \pmb { x } } _ { t } ) ) } \end{array}$ , where $\pmb { x } _ { ( i ) } ^ { * } = \mathrm { a r g m a x } _ { \pmb { x } \in \mathcal { X } } f ^ { ( i ) } ( \pmb { x } )$ . Then, our MC estimator for $\mathrm { B R } _ { T } ( \pmb { \theta } )$ is

$$
\widehat { \mathrm { B R } } _ { T } ( \pmb \theta ) : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } R _ { T } ^ { ( i ) } .\tag{2}
$$

As a result, we solve the following optimization problem instead of (1):

$$
\operatorname* { m i n } _ { \pmb { \theta } \in \Theta } \widehat { \mathrm { B R } } _ { T } ( \pmb { \theta } ) .\tag{3}
$$

This objective function $\widehat { \mathrm { B R } } _ { T } ( \pmb { \theta } )$ is a complicated black-box function for which analytical representation and derivative are dificult to obtain. Further, to evaluate $\widehat { \mathrm { B R } } _ { T } ( \pmb { \theta } )$ for a given $\theta ,$ we need to run a T-iteration BO for N times, performing exhaustive or massive evaluations is not practical.

```latex
Algorithm 2: Pre-trained BO
1 Generate N sample paths $\overline { { f ^ { ( 1 ) } , \ldots , f ^ { ( N ) } } }$ from $\mathcal { G P } ( \mu _ { 0 } , \Sigma _ { 0 } )$
2 Create initial $\mathcal { D } ^ { \mathrm { o u t e r } }$
3 for $i = 1 , \ldots , T ^ { \mathrm { o u t e r } }$ do
4 Fit a GP to D<sup>outer</sup>
5 $\pmb \theta \gets \mathrm { a r g m a x } _ { \pmb \theta \in \Theta } a ^ { \mathrm { o u t e r } } ( \pmb \theta )$
6 for $j = 1 , \dotsc , \bar { N }$ do
7 Evaluate $R _ { T } ^ { ( j ) }$ by running BO $( f ^ { ( j ) } , \theta , T )$
8 end
9 $\begin{array} { r } { \widehat { \mathrm { B R } _ { T } } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } R _ { T } ^ { ( j ) } } \end{array}$
10 ${ \mathcal { D } } ^ { \mathrm { o u t e r } } \gets { \mathcal { D } } ^ { \mathrm { o u t e r } } \cup \{ ( \pmb { \theta } , \widehat { \mathrm { B R } } _ { T } ) \}$
11 end
```

Therefore, we also apply BO to the optimization of (3). We call the optimization of θ outer BO and the optimization of x inner BO.

Algorithm 2 shows the procedure of outer BO. A dataset for outer BO is a set of $( \pmb { \theta } , \widehat { \mathrm { B R } } _ { T } ( \pmb { \theta } ) )$ , denoted as $\mathcal { D } ^ { \mathrm { o u t e r } }$ . By fitting a GP to $\widehat { \mathrm { B R } } _ { T } ( \pmb { \theta } )$ , an acquisition function for outer BO $a ^ { \mathrm { o u t e r } } ( \pmb { \theta } )$ can be defined. By performing inner BO N times, the cumulative regret $R _ { T } ^ { ( j ) }$ for the j-th sample path is evaluated at line 7. Although $\begin{array} { r } { \widehat { \mathrm { B R } } _ { T } ( \pmb { \theta } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } R _ { T } ^ { ( i ) } } \end{array}$ is optimized here, we can replace $R _ { T } ^ { ( i ) }$ with $- \textstyle \sum _ { t = 1 } ^ { T } f ^ { ( i ) } ( \hat { \pmb x } _ { t } )$ . This replacement only shifts $\mathrm { \widehat B R } _ { T }$ with a constant $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } T f ^ { ( i ) } ( \pmb { x } _ { ( i ) } ^ { * } ) } \end{array}$ , and it does not afect the optimal θ, while we can omit the calculation of $\boldsymbol { \mathbf { \mathit { x } } } _ { ( i ) } ^ { * }$ by the replacement. The same essential conclusion holds for the following theoretical analysis, regardless of which version is considered.

## 3.3 Theoretical Properties

We first consider the convergence of the MC estimator (2). In general, the approximation error of the MC estimator is typically $O ( 1 / \sqrt { N } )$ under some regularity condition such as the finite variance (e.g., Glasserman, 2004). Since a value of $f$ that follows a GP is not finitely bounded, it is not immediately evident whether the variance of $\mathrm { B R } _ { T }$ is finite. The following proposition guarantees the convergence of $\widehat { \mathrm { B R } } _ { T }$ :

Proposition 3.1. Assume that the kernel function is continuous, X is compact, and there exist constants $L < \infty$ and $\alpha > 0$ that satisfy

$$
d ( \pmb { x } , \pmb { x } ^ { \prime } ) : = \sqrt { \mathbb { E } [ \{ f ( \pmb { x } ) - f ( \pmb { x } ^ { \prime } ) \} ^ { 2 } ] } \leq L \| \pmb { x } - \pmb { x } ^ { \prime } \| _ { 2 } ^ { \alpha }\tag{4}
$$

for any suficiently small $\| { \pmb x } - { \pmb x } ^ { \prime } \| _ { 2 }$ . Then, for a given $\pmb \theta _ { i }$

$$
\mathbb { E } \left[ \left. \left( \widehat { \mathrm { B R } } _ { T } ( \pmb { \theta } ) - \mathrm { B R } _ { T } ( \pmb { \theta } ) \right) ^ { 2 } \right| \mathcal { D } _ { 0 } \right] \leq \frac { 4 C _ { \mathcal { D } _ { 0 } } T ^ { 2 } } { N }
$$

and

$$
\widehat { \mathrm { B R } } _ { T } ( \pmb \theta ) - \mathrm { B R } _ { T } ( \pmb \theta ) = O _ { p } \left( \frac { T } { \sqrt { N } } \right) ,
$$

where $\begin{array} { r } { C _ { \mathcal { D } _ { 0 } } : = \mathbb { E } \left[ \left. \left( \operatorname* { s u p } _ { \pmb { x } \in \mathcal { X } } \left| f ( \pmb { x } ) \right| \right) ^ { 2 } \right| \mathcal { D } _ { 0 } \right] < \infty . } \end{array}$

The proof is in Appendix D. The metric $d ( { \pmb x } , { \pmb x } ^ { \prime } )$ in the assumption (4) is called canonical pseudometric (e.g., Adler and Taylor, 2007), and Appendix E shows Gaussian kernel and Mat´ern kernel satisfy this condition. This proposition shows that the estimator $\widehat { \mathrm { B R } } _ { T }$ has a finite variance, and, as a consequence, the stochastic order is derived as $O _ { p } ( T / \sqrt { N } )$ . The dependence on N is same as the standard MC estimator, and on T is the first order. In this proposition, since we do not assume any specific acquisition function of inner BO, the result is generally applicable to a variety of BO algorithms.

For the case of the finite candidate Θ, proposition 3.1 derives the following corollary:

Corollary 3.1. Let Θ be a finite set of candidate parameters. Define

$$
\theta ^ { \star } \in \underset { \theta \in \Theta } { a r g m i n } \mathrm { B R } _ { T } ( \pmb \theta ) , \qquad \widehat { \pmb \theta } _ { N } \in \underset { \theta \in \Theta } { a r g m i n } \widehat { \mathrm { B R } } _ { T } ( \pmb \theta )
$$

as the optimizers of the true and estimated cumulative regret, respectively. Then

$$
\mathrm { B R } _ { T } ( \widehat { \pmb { \theta } } _ { N } ) - \mathrm { B R } _ { T } ( \pmb { \theta } ^ { \star } ) = O _ { p } \left( \frac { T } { \sqrt { N } } \right) .
$$

The proof is in Appendix F. Proposition 3.1 only shows the MC estimation error on a fixed θ. This corollary guarantees the diference between the expected cumulative regret of the true optimal configuration $\mathrm { B R } _ { T } ( \pmb { \theta } ^ { \star } )$ and those of achieved by estimated ${ \widehat { \pmb { \theta } } } _ { N }$ defined by the maximizer of the MC estimator $\widehat { \mathrm { B R } } _ { T }$ . For a continuous parameter space, the pointwise convergence in Proposition 3.1 does not directly extend Corollary 3.1, since it requires uniform convergence over Θ, i.e., $\begin{array} { r } { \operatorname* { s u p } _ { \pmb { \theta } \in \Theta } \left| \widehat { \mathrm { B R } } _ { T , N } ( \pmb { \theta } ) - \mathrm { B R } _ { T } ( \pmb { \theta } ) \right| = O _ { p } \left( \frac { T } { \sqrt { N } } \right) } \end{array}$ , for which additional regularity conditions may be required for the continuous case. We leave a complete analysis of the continuous case for future work.

## 4 Extensions

## 4.1 Multi-fidelity Approach for Outer BO

While our optimization target is the T iteration cumulative regret (2), the cumulative regret with earlier $t < T$ iterations can be seen as a cheaper approximation of (2). It is possible to regard the optimization problem (3) as a multi-fidelity optimization problem, in which $\widehat { \mathrm { B R } } _ { T } ( \pmb { \theta } )$ is the highest fidelity function and $\widehat { \mathrm { B R } } _ { t } ( \pmb \theta )$ for $t < T$ is a lower fidelity function. Therefore, we can exploit multi-fidelity BO (MFBO) to accelerate the optimization (3).

A variety of methods have been proposed for MFBO (e.g., Huang et al., 2006; Kandasamy et al., 2016; Takeno et al., 2020), in which a pair of an input point and its fidelity level is selected at each iteration. Here, we consider the case that there exists only one low fidelity level is defined by the $T _ { \mathrm { l o w } } < T$ iteration inner BO (actually, it is possible to create any H fidelity levels $T _ { 1 } < T _ { 2 } < \cdots <$ $T _ { H } = T )$ . In a typical setting of MFBO, a correlated multi-output GP (Alvarez et al., 2012) is<sup>´</sup> constructed to capture the relation among diferent fidelity levels. In our case, a two-output GP represents $\widehat { \mathrm { B R } } _ { t } ( \pmb \theta )$ for $t = T _ { \mathrm { l o w } }$ and T. MFBO can select a pair of θ and its high $( t = T )$ or low $( t = T _ { \mathrm { l o w } } )$ fidelity level based on an MFBO acquisition function.

## 4.2 Optimizing Multi-Stage Configurations

We have set a fixed θ during inner BO. On the other hand, θ can be changed during inner BO, by which altering the search strategy at some iteration becomes possible. For brevity, we here assume that we change θ only once at the iteration $T _ { \mathrm { m i d d l e } } ( < T )$ . Let $\pmb \theta _ { ( 1 ) }$ be the parameter of earlier $T _ { \mathrm { m i d d l e } }$ iterations and $\pmb \theta _ { ( 2 ) }$ be that of later iterations from $T _ { \mathrm { m i d d l e } } \dot { + } \dot { 1 }$ to $T ,$ . Then, by re-defining $\pmb { \theta } = ( \pmb { \theta } _ { ( 1 ) } ^ { \top } , \pmb { \theta } _ { ( 2 ) } ^ { \top } ) ^ { \top }$ , we can optimize both parameters simultaneously by minimizing $\widehat { \mathrm { B R } } _ { T } ( \pmb { \theta } )$ though outer BO. In this case, MFBO can also be applied by regarding $\widehat { \mathrm { B R } } _ { T _ { \mathrm { m i d d l e } } } ( \pmb { \theta } )$ and $\widehat { \mathrm { B R } } _ { T } ( \pmb { \theta } )$ as low and high fidelity, respectively $( \mathrm { i . e . , ~ } T _ { \mathrm { l o w } } = T _ { \mathrm { m i d d l e } } )$ . However, unlike $\widehat { \mathrm { B R } } _ { T } ( \pmb \theta ) , \widehat { \mathrm { B R } } _ { T _ { \mathrm { m i d d l e } } } ( \pmb \theta )$ only depends on $\pmb \theta _ { ( 1 ) }$ and does not depend on $\theta _ { ( 2 ) }$ . Thus, the lower fidelity function can be written as $\widehat { \mathrm { B R } } _ { T _ { \mathrm { m i d d l e } } } ( \pmb { \theta } _ { ( 1 ) } )$ , i.e., low and high fidelity function have diferent input dimensions. To apply MFBO, this relation should be incorporated into the surrogate model.

We use a model inspired by the autoregressive co-kriging model (Kennedy and O’Hagan, 2000). The following definition can handle low and high fidelity functions having diferent input dimensions, while modeling their correlated structure:

$$
\begin{array} { r l } & { \widehat { \mathrm { B R } } _ { t } ( \pmb \theta _ { ( 1 ) } ) \approx f ^ { ( \mathrm { l o w } ) } ( \pmb \theta _ { ( 1 ) } ) , } \\ & { \quad \widehat { \mathrm { B R } } _ { T } ( \pmb \theta ) \approx f ^ { ( \mathrm { h i g h } ) } ( \pmb \theta _ { ( 1 ) } , \pmb \theta _ { ( 2 ) } ) : = f ^ { ( \mathrm { l o w } ) } ( \pmb \theta _ { ( 1 ) } ) + g ( \pmb \theta _ { ( 1 ) } , \pmb \theta _ { ( 2 ) } ) , } \end{array}
$$

where $f ^ { ( \mathrm { l o w } ) }$ and $g$ follow independent $\mathrm { G P s }$ and $\rho$ is a . This model enables us to evaluate the posterior from the dataset containing both $\widehat { \mathrm { B R } } _ { T _ { \mathrm { m i d d l e } } } ( \pmb { \theta } _ { ( 1 ) } )$ and $\widehat { \mathrm { B R } } _ { T } ( \pmb { \theta } )$ . See Appendix A for detail.

## 5 Related Work

Learning optimization policies. Our work is closely related to learning to optimize (L2O) and meta-learning, where optimization algorithms are learned from a collection or distribution of optimization problems. L2O aims to learn optimization procedures from their performance on training problems so that they generalize to new problems drawn from a similar distribution (Chen et al., 2022). For black-box optimization, Chen et al. (2017) learn a derivative-free optimizer parameterized by a recurrent neural network using training functions sampled from function distributions, including GPs, while TV et al. (2019) similarly learn a recurrent black-box optimizer with a regret-based meta-objective. More specifically in the context of BO, MetaBO (Volpp et al., 2020) meta-learns an acquisition function using reinforcement learning on related tasks, yielding a BO policy specialized to a given task distribution, while FSAF (Hsieh et al., 2021) combines reinforcement learning and meta-learning to learn acquisition strategies that can rapidly adapt to new black-box functions. Related ideas have also been explored more broadly in meta-black-box optimization, where the search strategy itself is learned across optimization problems. For example, Lange et al. (2023) meta-learn evolution strategies parameterized by attention-based models, and more recent work has investigated pre-training optimization models for transfer to unseen black-box optimization tasks (Li et al., 2024) and learning neural optimizers for black-box optimization under limited evaluation budgets (Li et al., 2025). Our approach shares with these methods the general principle of exploiting a distribution of optimization problems to improve optimization performance on future tasks. However, rather than learning an optimizer or acquisition policy itself, we retain the structure of an established BO algorithm and optimize only its pre-specified algorithmic param eters. By restricting the search to a structured class of established BO algorithms, our approach is designed to leverage knowledge accumulated in the BO literature, including established acquisition rules and their principles for balancing exploration and exploitation, while still allowing adaptation to the target problem distribution.

Learning distributions over objective functions. A complementary line of work uses data from related tasks to learn surrogate models or prior distributions that can transfer information to new optimization tasks. HyperBO (Wang et al., 2024) pre-trains a GP prior using data from related functions and applies the learned prior to BO on new tasks. More generally, learned stochasticprocess models such as Neural Processes (Garnelo et al., 2018) learn distributions over functions from collections of related functions and condition their predictions on observations from a new task. While these approaches focus on constructing a transferable surrogate model or distribution over objective functions from past data, our work addresses how to configure a BO algorithm given such a distribution. The two directions are therefore complementary. For example, a pre-trained GP prior can directly serve as the distribution over objective functions in our framework, from which sample paths are generated to evaluate and tune BO parameters. More generally, our framework can be combined with probabilistic surrogate models from which suitable function samples can be generated.

## 6 Experiments

We empirically evaluate the performance of the proposed framework. We refer to the proposed method as Pretrained BO and to its multi-stage extension described in Section 4.2 as Pretrained BO-MS. For MFBO in outer BO, we set $T _ { \mathrm { l o w } } = T / 2$ and use MFSKO (Huang et al., 2006) as an acquisition function. We use the configuration of TuRBO (Eriksson et al., 2019) and LSBO (G´omez Bombarelli et al., 2018) for demonstrating the performance of our framework.

## 6.1 TuRBO

We first consider the optimization of GP sample-path functions using TuRBO. For Pretrained BO, we optimize seven TuRBO parameters, including thresholds for the numbers of successes and failures and the number of trust regions. The resulting configuration space contains 3,888 discrete candidates. For Pretrained BO-MS, using separate configurations for the first and second stages doubles the number of parameters. To reduce the computational cost, we therefore restrict the configuration to three parameters for each stage. Each stage has 105 candidate configurations, resulting in $1 0 5 ^ { 2 } = 1 1$ ,025 multi-stage candidates. See Appendix B.1 for further details.

We use a zero prior mean and a Gaussian kernel for the GP, with observation-noise variance $\sigma _ { \epsilon } ^ { 2 } = 1 0 ^ { - 6 }$ . The number of inner-BO iterations is set to $T ~ = ~ 2 0 0$ . We first randomly select $n _ { 0 } = 1 0 0$ input points from X and construct the initial dataset by evaluating a benchmark function at these points. We consider Sphere, Rosenbrock, and Levy as benchmark functions (Surjanovic and Bingham, 2013). From the GP posterior constructed using this initial dataset, we generate both the training sample paths $f ^ { ( 1 ) } , \ldots , f ^ { ( N ) }$ used by Pretrained BO and test functions $f ^ { \mathrm { t e s t } }$ used to evaluate the performance of TuRBO with the estimated configuration ${ \hat { \theta } } _ { N }$ . We independently generate ten test functions and report the average performance over them. For outer BO, we also use a Gaussian kernel. The number of outer-BO iterations is set to $T _ { \mathrm { o u t e r } } = 2 0 0$

Figure 1 shows the results. As baselines, we include MetaBO (from https://github.com/ boschresearch/MetaBO, which is AGPL-3.0 license), standard BO with expected improvement (EI), and 1,000 randomly selected configurations. MetaBO is trained using the same training functions as the proposed method and learns an acquisition function from them. Standard BO (EI) uses the conventional EI acquisition function without pre-training. For the random-configuration baseline, 1,000 configurations are sampled from the candidate set Θ (the multi-stage setting is not applied to this baseline).

=(f) (d, initial data)  
![](images/899744a1edf279b4cc089bb8ab31c67967a5dbe3fa9257bee7ced15bd5baf8d6.jpg)  
(a) (d, initial data) = (5, Sphere)

![](images/cea7d4ca5f5abc7c2af72a6f16d4a3f41ade69c678ea190718ea52caa802c314.jpg)  
(b) (d, initial data) = (10, Sphere)

![](images/b2fb676cb220a31cede79c793709e87e9e1b59d0b8168b500e4208d307034cc8.jpg)  
(c) (d, initial data) = (20, Sphere)

![](images/e17ff41ad367199329a52d5a63776b6d8a806e9a660a1d955a03751b8388f8cc.jpg)  
(d) (d, initial data)=(5, Rosenbrock)(e)

![](images/42cd8fa28908e178a4c46a3569b2de89441e9d3f45c94bf48fee88820f11c2ea.jpg)  
(d, initial data)  
(10, Rosenbrock)

![](images/83edafb40534863cc8c68c4cbfb2e637243e10c090ad0872da6cecfdc5d0b5e2.jpg)  
(20, Rosenbrock)

![](images/66fa94b51f5715c1d6b6d27fc25c66669afc620453575b5597b1026b8814593a.jpg)  
(g) (d, initial data) = (5, Levy)

![](images/72d2cca481a3264c241582668b2022cda92b63dae125ce72c2766577a105decd.jpg)  
(h) (d, initial data) = (10, Levy)

![](images/b151827aac1fb752ddce595f2309cc7169164bb25612c6d9cbd8f3b47495916b.jpg)  
(i) (d, initial data) = (20, Levy)  
Figure 1: Comparison of test function performance.

We first compare Pretrained BO with the 1,000 randomly selected configurations. In many settings, the black dashed curve corresponding to Pretrained BO remains below the collection of 1,000 gray curves. Since these configurations are sampled uniformly from the candidate set Θ, when the Pretrained BO curve is below all 1,000 random configurations, its selected configuration can roughly be regarded as belonging to the top 0.1% of the candidate set in terms of performance. Pretrained BO also outperforms standard BO and MetaBO in many of the settings. Even in settings such as Figs. 1 (d) and (f), where such an improvement over these existing methods is not observed, comparison with the gray curves indicates that Pretrained BO still identifies a highperforming configuration within Θ. Pretrained BO-MS yields substantial additional improvements in several settings, while in others, such as Figs. 1 (e) and (h), its performance is comparable to that of Pretrained BO without the multi-stage extension. One possible reason is that, for computational

![](images/6a6da9b69a1cbcd02d6bb984440aad9141d21c6c37e0f03151a63bf9a44d9b15.jpg)  
(a) Pretrained BO

![](images/4c6835208894ac73065a1632cecdf552cf004cecf430dac6f2a7289d0389b4d7.jpg)  
(d, initial data) = (10, Sphere)  
(b) Pretrained BO

![](images/65fde71221d72390b20de72e7288d68ad0f24995b3513f46d8ab129861dea41e.jpg)  
(c) Pretrained BO  
(d, initial data) = (10, Levy)

(d, initial data) = (10, Rosenbrock)  
![](images/a482acf4f13097986f228a9e8db3e0ad6aebdace262f3df3ac89e27dd68e538e.jpg)  
(d) Pretrained BO-MS  
(d, initial data) = (10, Sphere)

![](images/4350f7bd329b2d89798a2acd6123e38dfe5243ebb5eefb731ded0bcb181e9b88.jpg)  
(e) Pretrained BO-MS

![](images/7fb3a26ab46bbe1e007e4bb2b07fe16be0c5353759d1057655a7844fb24e1fca.jpg)  
(d, initial data) = (10, Rosenbrock)  
(f) Pretrained BO-MS  
(d, initial data) = (10, Levy)

Figure 2: Transition of $\begin{array} { r } { \frac { 1 } { N } \sum _ { t = 1 } ^ { T } f ^ { ( i ) } ( \pmb { x } _ { t } ) } \end{array}$ . (a)-(c): Pretrained BO, (d)-(f): Pretrained BO-MS

![](images/ce2c8436eb106df726ee8625792aa53a5d4327fec1c85aad31293cc4a4376810.jpg)  
Figure 3: Score transition of LSBO.

tractability, we reduced the dimensionality of the configuration space for Pretrained BO-MS. The results with $n _ { 0 } = 5 0 $ and $n _ { 0 } = 3 0 0$ are in Appendix C.1, and those with std error plots are in Appendix C.2.

Figure 2 illustrates the optimization process of outer BO. The vertical axis shows $\begin{array} { r } { \frac { 1 } { N } \sum _ { t = 1 } ^ { T } f ^ { ( i ) } ( \pmb { x } _ { t } ) } \end{array}$ and the horizontal axis represents the total evaluation cost. For each setting, we compare standard BO, MFBO (MFSKO), and random search for optimizing θ (Note that in Fig. 1, standard BO was used for Pretrained BO, whereas MFBO was used for Pretrained BO-MS. See Appendix B.3 for detail of MFBO setting here). Figure 2 shows a general tendency for BO to search more eficiently than random search and for MFBO to further improve the search eficiency over standard BO.

## 6.2 LSBO

We next consider a synthetic latent-space optimization problem based on an image-classification CNN. This setting is inspired by Boyar and Takeuchi (2024). The CNN is pre-trained on Fashion-

MNIST (Xiao et al., 2017), which is released under the MIT License. We assign a score to each Fashion-MNIST class and aim to generate images with high scores. Let $s _ { y }$ denote the score assigned to class y, and let $p _ { 1 } , \ldots , p _ { K }$ denote the class probabilities predicted by the CNN, where K is the number of classes. The expected score of an image is then defined as $\sum { } _ { y \in [ K ] } p _ { y } s _ { y } .$

We first train a VAE to construct a 20-dimensional latent space and then perform BO in this latent space to maximize the expected score. Specifically, BO selects a point in the latent space, the point is decoded into an image by the VAE, and the objective value is obtained by evaluating the expected score of the decoded image using the CNN. This setup can be viewed as a simplified analogue of, for example, searching a molecular latent space for compounds with desirable properties.

We treat the VAE hyperparameters as the configuration θ. Specifically, we optimize three weighting parameters controlling properties of the latent representation (Chen et al., 2018), such as independence, together with the dimensionality of the latent embedding, resulting in four configuration parameters. We set the inner-BO horizon to $T = 1 0 0$ . The training functions are generated by perturbing the class scores $s _ { y }$ . Further details are provided in Appendix B.2.

Figure 3 shows the results, where the vertical axis represents the expected score. For clarity, we show only Pretrained BO-MS and the results for 500 randomly selected configurations. The proposed method identifies a configuration with high optimization performance among the candidate configurations.

## 7 Conclusion

We proposed Pretrained BO for configuring a BO algorithm by minimizing its expected cumulative regret estimated from GP sample paths. We established theoretical guarantees for the MC approximation and extended the framework to multi-fidelity and multi-stage settings. Experiments on TuRBO and LSBO demonstrated that the proposed method can identify high-performing configurations. Our results suggest that prior knowledge about objective functions can be exploited not only for surrogate modeling but also for configuring the BO algorithm itself. A limitation is that misspecification of the estimated GP posterior may lead to suboptimal BO parameter choices. Evaluating the robustness under such model misspecification is an important direction for future work.

## Acknowledgement

This work was partially supported by MEXT KAKENHI (25K03182), MEXT Supporting Pioneering Re-search through AI for 1,000 Discovery challenges Program (SPReAD) Japan Grant Number JPMXP1726275243, and Data Creation and Utilization Type Material Research and Development Project (Grant No. JP-MXP1122712807) of MEXT.

## References

Adler, R. J. and Taylor, J. E. (2007). Random Fields and Geometry. Springer Monographs in Mathematics. Springer New York, NY.

Alvarez, M. A., Rosasco, L., and Lawrence, N. D. (2012). Kernels for vector-valued functions: A <sup>´</sup> review. Foundations and Trends® in Machine Learning, 4(3):195–266.

Boyar, O. and Takeuchi, I. (2024). Latent space bayesian optimization with latent data augmentation for enhanced exploration. Neural Computation, 36(11):2446–2478.

Brochu, E., Cora, V. M., and de Freitas, N. (2010). A tutorial on bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning. arXiv preprint arXiv:1012.2599.

Chen, R. T. Q., Li, X., Grosse, R., and Duvenaud, D. (2018). Isolating sources of disentanglement in vaes. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS’18, page 2615–2625, Red Hook, NY, USA. Curran Associates Inc.

Chen, T., Chen, X., Chen, W., Heaton, H., Liu, J., Wang, Z., and Yin, W. (2022). Learning to optimize: A primer and a benchmark. Journal of Machine Learning Research, 23(189):1–59.

Chen, Y., Hofman, M. W., Colmenarejo, S. G., Denil, M., Lillicrap, T. P., Botvinick, M., and de Freitas, N. (2017). Learning to learn without gradient descent by gradient descent. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 748–756. PMLR.

Deringer, V. L., Bart´ok, A. P., Bernstein, N., Wilkins, D. M., Ceriotti, M., and Cs´anyi, G. (2021). Gaussian process regression for materials and molecules. Chemical Reviews, 121(16):10073–10141.

Eriksson, D., Pearce, M., Gardner, J. R., Turner, R., and Poloczek, M. (2019). Scalable global optimization via local Bayesian optimization. Curran Associates Inc., Red Hook, NY, USA.

Garnelo, M., Schwarz, J., Rosenbaum, D., Viola, F., Rezende, D. J., Eslami, S. M. A., and Teh, Y. W. (2018). Neural processes. In ICML Workshop on Theoretical Foundations and Applications of Deep Generative Models.

Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering, volume 53 of Applications of Mathematics. Springer, New York.

G´omez-Bombarelli, R., Wei, J. N., Duvenaud, D., Hern´andez-Lobato, J. M., S´enchez-Lengeling, B., Sheberla, D., Aguilera-Iparraguirre, J., Hirzel, T. D., Adams, R. P., and Aspuru-Guzik, A. (2018). Automatic chemical design using a data-driven continuous representation of molecules. ACS Central Science, 4(2):268–276.

Hsieh, B.-J., Hsieh, P.-C., and Liu, X. (2021). Reinforced few-shot acquisition function learning for bayesian optimization. In Advances in Neural Information Processing Systems, volume 34.

Huang, D., Allen, T., Notz, W., and Miler, R. (2006). Sequential kriging optimization using multiple-fidelity evaluations. Structural and Multidisciplinary Optimization, 32(5):369–382.

Kandasamy, K., Dasarathy, G., Oliva, J., Schneider, J., and P´oczos, B. (2016). Gaussian process bandit optimisation with multi-fidelity evaluations. In Proceedings of the 30th International Conference on Neural Information Processing Systems, NIPS’16, page 1000–1008, Red Hook, NY, USA. Curran Associates Inc.

Kennedy, M. C. and O’Hagan, A. (2000). Predicting the output from a complex computer code when fast approximations are available. Biometrika, 87(1):1–13.

Kingma, D. P. and Welling, M. (2014). Auto-encoding variational bayes. In International Conference on Learning Representations (ICLR).

Lange, R. T., Schaul, T., Chen, Y., Zahavy, T., Dalibard, V., Lu, C., and Singh, S. (2023). Discovering evolution strategies via meta-black-box optimization. In International Conference on Learning Representations.

Li, Y., Zhang, S., Liu, J., and Tan, K. C. (2024). Pretrained optimization model for zero-shot black box optimization. In Advances in Neural Information Processing Systems, volume 37.

Li, Y., Zhang, S., Liu, J., and Tan, K. C. (2025). B2Opt: Learning to optimize black-box optimization with little budget. In Proceedings of the AAAI Conference on Artificial Intelligence.

Rahimi, A. and Recht, B. (2008). Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems 20, pages 1177–1184. Curran Associates, Inc.

Srinivas, N., Krause, A., Kakade, S., and Seeger, M. (2010). Gaussian process optimization in the bandit setting: No regret and experimental design. In Proceedings of the 27th International Conference on International Conference on Machine Learning, pages 1015–1022. Omnipress.

Stein, M. L. (1999). Interpolation of Spatial Data: Some Theory for Kriging. Springer Series in Statistics. Springer, New York.

Surjanovic, S. and Bingham, D. (2013). Virtual library of simulation experiments: Test functions and datasets. Retrieved February 4, 2024, from http://www.sfu.ca/<sub>\~</sub>ssurjano.

Takeno, S., Fukuoka, H., Tsukada, Y., Koyama, T., Shiga, M., Takeuchi, I., and Karasuyama, M. (2020). Multi-fidelity Bayesian optimization with max-value entropy search and its parallelization. In Proceedings of the 37th International Conference on Machine Learning, volume 119, pages 9334–9345. PMLR.

TV, V., Malhotra, P., Vig, L., and Shrof, G. (2019). Meta-learning for black-box optimization. arXiv preprint arXiv:1907.06901.

Vershynin, R. (2018). High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge.

Volpp, M., Fr¨ohlich, L. P., Fischer, K., Doerr, A., Falkner, S., Hutter, F., and Daniel, C. (2020). Meta-learning acquisition functions for transfer learning in bayesian optimization. In International Conference on Learning Representations.

Wang, X., Jin, Y., Schmitt, S., and Olhofer, M. (2023). Recent advances in bayesian optimization. ACM Comput. Surv., 55(13s).

Wang, Z., Dahl, G. E., Swersky, K., Lee, C., Nado, Z., Gilmer, J., Snoek, J., and Ghahramani, Z. (2024). Pre-trained gaussian processes for bayesian optimization. Journal of Machine Learning Research, 25(212):1–83.

Wu, Y. (2020). Lecture notes on information-theoretic methods for high-dimensional statistics. http://www.stat.yale.edu/ yw562/teaching/it-stats.pdf.

Xiao, H., Rasul, K., and Vollgraf, R. (2017). Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. CoRR, abs/1708.07747.

# Supplementary Appendix of Pre-training of Bayesian Optimization Algorithm through Bayesian Optimization

## A Details of MFBO for Multi-Stage Configurations

$$
\begin{array} { r l } & { f ^ { \mathrm { l o w } } \sim \mathcal { G P } ( 0 , k _ { l } ( \pmb { \theta } _ { ( 1 ) } , \pmb { \theta } _ { ( 1 ) } ^ { \prime } ) ) , } \\ & { \quad g \sim \mathcal { G P } ( 0 , k _ { g } ( ( \pmb { \theta } _ { ( 1 ) } , \pmb { \theta } _ { ( 2 ) } ) , \ ( \pmb { \theta } _ { ( 1 ) } ^ { \prime } , \pmb { \theta } _ { ( 2 ) } ^ { \prime } ) ) ) , } \end{array}
$$

where $k _ { l }$ and $k _ { g }$ are kernel functions. From the definition of $f ^ { ( \mathrm { l o w } ) }$ and $f ^ { \mathrm { ( h i g h ) } }$

$$
\begin{array} { r l } & { \qquad \mathrm { C o v } ( f ^ { ( \mathrm { l o w } ) } ( \pmb { \theta } _ { ( 1 ) } ) , ~ f ^ { ( \mathrm { l o w } ) } ( \pmb { \theta } _ { ( 1 ) } ^ { \prime } ) ) = k _ { l } ( \pmb { \theta } _ { ( 1 ) } , \pmb { \theta } _ { ( 1 ) } ^ { \prime } ) , } \\ & { \qquad \mathrm { C o v } ( f ^ { ( \mathrm { l o w } ) } ( \pmb { \theta } _ { ( 1 ) } ) , ~ f ^ { \mathrm { ( h i g h ) } } ( \pmb { \theta } _ { ( 1 ) } ^ { \prime } , \pmb { \theta } _ { ( 2 ) } ^ { \prime } ) ) = k _ { l } ( \pmb { \theta } _ { ( 1 ) } , \pmb { \theta } _ { ( 1 ) } ^ { \prime } ) , } \\ & { \qquad \mathrm { C o v } ( f ^ { ( \mathrm { h i g h } ) } ( \pmb { \theta } _ { ( 1 ) } , \pmb { \theta } _ { ( 2 ) } ) , ~ f ^ { \mathrm { ( h i g h ) } } ( \pmb { \theta } _ { ( 1 ) } ^ { \prime } , \pmb { \theta } _ { ( 2 ) } ^ { \prime } ) ) = k _ { l } ( \pmb { \theta } _ { ( 1 ) } , \pmb { \theta } _ { ( 1 ) } ^ { \prime } ) + k _ { g } ( ( \pmb { \theta } _ { ( 1 ) } , \pmb { \theta } _ { ( 2 ) } ) , ~ ( \pmb { \theta } _ { ( 1 ) } ^ { \prime } , \pmb { \theta } _ { ( 2 ) } ^ { \prime } ) ) . } \end{array}
$$

Let $\tilde { \boldsymbol { \theta } }$ be $\pmb \theta _ { ( 1 ) }$ for the low fidelity input and θ for the high fidelity input. Then, the kernel for multi-stage observation is

$$
\begin{array} { r } { k _ { \mathrm { m s } } ( \widetilde { \pmb { \theta } } , \widetilde { \pmb { \theta } } ^ { \prime } ) = \left\{ \begin{array} { l l } { k _ { l } ( \pmb { \theta } _ { ( 1 ) } , \pmb { \theta } _ { ( 1 ) } ^ { \prime } ) + k _ { g } ( ( \pmb { \theta } _ { ( 1 ) } , \pmb { \theta } _ { ( 2 ) } ) , \ ( \pmb { \theta } _ { ( 1 ) } ^ { \prime } , \pmb { \theta } _ { ( 2 ) } ^ { \prime } ) ) , } & { \mathrm { ~ i f ~ b o t h ~ o f ~ } \widetilde { \pmb { \theta } } \mathrm { ~ a n d ~ } \widetilde { \pmb { \theta } } ^ { \prime } \mathrm { ~ a r e ~ f r o m ~ h i g h ~ f i d e l i t y } } \\ { k _ { l } ( \pmb { \theta } _ { ( 1 ) } , \pmb { \theta } _ { ( 1 ) } ^ { \prime } ) , } & { \mathrm { ~ o t h e r w i s e } } \end{array} \right. } \end{array}
$$

The acquisition function of MFSKO (Huang et al., 2006) α(θ, fidelity) requires full θ including $\pmb \theta _ { ( 2 ) }$ even when the acquisition function value for fidelity = low is calculated. To avoid this issue, we first optimize $( \bar { \theta } _ { ( 1 ) } , \bar { \theta } _ { ( 2 ) } ) = \mathrm { a r g m a x } _ { \theta } \alpha ( \theta , \mathrm { f i d e l i t y \nonumber = \ h i g h } )$ . Next, by fixing $\bar { \pmb { \theta } } _ { ( 2 ) }$ , we optimize $\operatorname* { m a x } _ { \pmb { \theta } _ { ( 1 ) } } \alpha ( [ \pmb { \theta } _ { ( 1 ) } , \bar { \pmb { \theta } } _ { ( 2 ) } ]$ , fidelity = low). The same procedure is applicable to MFMES (Takeno et al., 2020).

## B Experimental Detail

## B.1 Setting of TuRBO Experiment

In Pretrained BO, Θ is defined as follows

• Success/failure tolerance $\in [ 1 , 2 , 3 , 4 ]$ : The number of consecutive successes/failures required to expand/shrink the trust region.

• Initial TR size ∈ [0.8, 1.0, 1.2, 1.4]: The initial side length of the trust region.

• Maximum TR size $\in [ 2 , 4 , 8 ]$ : The maximum allowed side length of the trust region.

• Minimum TR size $\in [ 0 . 0 1 , 0 . 1 , 0 . 5 ]$ : The minimum allowed side length of the trust region.

• Number of $\mathrm { T R s } \in [ 1 , 3 , 5 ]$ : The number of trust regions maintained in parallel.

• Expansion factor $\in [ 1 . 5 , 2 , 3 ]$ : The factor by which the trust region is expanded after suficient successes.

• Shrinkage factor $\in [ 0 . 3 , 0 . 5 , 0 . 7 ]$ : The factor by which the trust region is shrunk after suficient failures.

For Pretrained BO-MS, candidates are success/failure tolerance $\in [ 1 , 2 , 3 , 4 , 5 ]$ , initial TR size $\in$ $[ 0 . 8 , 1 . 0 , 1 . 2 , 1 . 4 ]$ , and minimum TR size $\in [ 0 . 0 1 , 0 . 1 , 0 . 5 ]$

Let $v ^ { 2 }$ be the variance of initial $n _ { 0 }$ points. To generate initial sample path functions $f ^ { ( 1 ) } , \ldots , f ^ { ( N ) }$ parameters in Gaussian kernel are that the length scale was 0.5, the variance parameter was $0 . 3 \times v ^ { 2 }$ and the noise variance is $1 0 ^ { - 6 } \times v ^ { 2 }$

In outer BO, parameters of (Gaussian) kernel (length scale, variance parameter, noise variance, and correlation factor $\rho$ in MFGP) are optimized by marginal log-likelihood at every iteration.

In inner BO, parameters of (Gaussian) kernel (length scale, variance parameter, and noise variance) are optimized by marginal log-likelihood at every three iterations.

To generate training functions $\hat { f } ^ { ( 1 ) } , \ldots , \hat { f } ^ { ( N ) }$ in the continuous space, random Fourier feature (Rahimi and Recht, 2008) was used, in which the number of basis functions was set 1, 000.

## B.2 Setting of LSBO Experiment

We employ a VAE in which each component loss has its own weight parameter (Chen et al., 2018) by using the implementation https://github.com/rtqichen/beta-tcvae. Let $p _ { \mathrm { d a t a } } ( \pmb { x } )$ denote the empirical data distribution, $q ( \pmb { z } \mid \pmb { x } )$ the encoder distribution, and z the latent representation. The aggregated posterior is defined as

$$
q ( z ) = \mathbb { E } _ { p _ { \mathrm { d a t a } } ( \boldsymbol { x } ) } [ q ( \boldsymbol { z } \mid \boldsymbol { x } ) ] ,
$$

with $q ( z _ { j } )$ denoting its j-th marginal distribution. Following the ELBO decomposition of Chen et al. (2018), we consider the weighted objective

$$
\begin{array} { l } { \mathcal { L } _ { \alpha , \beta , \gamma } = \mathbb { E } _ { p _ { \mathrm { d a t a } } ( \boldsymbol { x } ) q ( \boldsymbol { z } | \boldsymbol { x } ) } [ \log p ( \boldsymbol { x } \mid \boldsymbol { z } ) ] - \alpha I _ { q } ( \boldsymbol { x } ; \boldsymbol { z } ) } \\ { \qquad - \beta D _ { \mathrm { K L } } ( q ( \boldsymbol { z } ) \| \displaystyle \prod _ { j } q ( \boldsymbol { z } _ { j } ) ) - \gamma \displaystyle \sum _ { j } D _ { \mathrm { K L } } ( q ( \boldsymbol { z } _ { j } ) \| p ( \boldsymbol { z } _ { j } ) ) , } \end{array}\tag{5}
$$

where the three coeficients control distinct properties of the latent representation:

• α controls the mutual information penalty between the data and the latent representation, regulating how much information about the data is retained in the latent representation.

• β controls the total correlation penalty, encouraging statistical independence among the latent dimensions and thereby promoting disentanglement.

• γ controls the dimension-wise KL penalty, encouraging the marginal distribution of each latent dimension to match its corresponding prior.

Θ consists of $\alpha , \beta , \gamma$ and the latent dimension. The finite candidate set is created in such a way that $\log _ { 1 0 } \alpha , \log _ { 1 0 } \beta$ and $\log _ { 1 0 } \gamma$ take from −2.0 to 2.0 with the uniform grid of 0.5. The candidate latent dimensions are $4 , 8 , 1 6 , \ldots , 3 2$ . Log EI is use in inner BO (LSBO).

We define the score $s _ { y } = \exp ( - 2 \times ( y - y _ { \mathrm { t g t } } ) ^ { 2 } )$ . This means that class $y _ { \mathrm { t g t } }$ (set as 4 in the experiments) has the highest score, and the scores of the other classes are determined by the distance measured by the class label. Since the class labels in Fashion-MNIST are not ordinal, this distance has no interpretation in the image space; we define it solely to create a setting in which each class is associated with a diferent level of preference. The perturbation of $s _ { y }$ to generate training functions are defined by $y _ { \mathrm { t g t } } + y _ { \epsilon }$ , where $y _ { \epsilon } \sim \mathcal { U } _ { [ - 0 . 5 , 0 . 5 ] }$

## B.3 Additional Details of MFBO

In Fig. 2, we evaluate BO and MFBO in Pretrained BO and Pretrained BO-MS, respectively. In Pretrained BO, standard multi-output GPs are applicable, because it shares the same input dimensions among diferent fidelities unlike Pretrained BO-MS. On the other hand, in Pretrained BO-MS, we need to use our model derived in Section 4.2, in which diferent input dimensions can be handled.

In Pretrained BO, this time, we employed linear model of coregionalization (LMC) (Alvarez<sup>´</sup> et al., 2012). Let $K ( \pmb \theta , \pmb \theta ^ { \prime } ) \in \mathbb R ^ { 2 \times 2 }$ be the covariance matrix of θ with low fidelity and $\pmb { \theta } ^ { \prime }$ with high fidelity. In LMC, it is set as

$$
K ( \theta , \theta ^ { \prime } ) = \sum _ { q = 1 } ^ { Q } B _ { q } k _ { q } ( \theta , \theta ^ { \prime } ) ,
$$

where $B _ { q } \in \mathbb { R } ^ { 2 \times 2 }$ is the coregionalization matrix. This means that the latent kernel $k _ { q }$ is shared across diferent fidelities. Specifically, we use the form $\begin{array} { r } { B _ { q } = { w _ { q } } { w _ { q } ^ { \top } } + \mathrm { d i a g } ( { k _ { q } } ) } \end{array}$ , where $\boldsymbol { w _ { q } } \in \mathbb { R } ^ { 2 }$ and $k _ { q } \in \mathbb { R } ^ { 2 }$ are parameters, which we estimate though marginal log-likelihood (this is rank-1 setting of IndexKernel of $\mathrm { G P y T o r c h } )$

## C Other Results

## C.1 Results on Diferent $n _ { 0 }$

Figures 4 and 5 are the same setting as $\mathrm { F i g . 1 }$ , except for that their initial points are 50 and 300.

## C.2 Results with Error Bars

Standard errors are shown in Figs. 6, 7, and 8. Note that each one of random 1000 configs has its standard error because the 10 runs were performed for each one of 1000 configs.

## D Proof of Proposition 3.1

In the proof, we use the following version of Dudley’s entropy bound and Borell–TIS inequality, adapted to our notation from (Adler and Taylor, 2007):

Theorem D.1 (Dudley’s entropy bound (adopted from Adler and Taylor, 2007) ). Let $f = \{ f ( { \pmb x } )$ $\pmb { x } \in \mathcal { X } \}$ be a centered separable Gaussian process, and define its canonical pseudometric $b y$

$$
d ( \pmb { x } , \pmb { x } ^ { \prime } ) : = \left( \mathbb { E } \left[ ( f ( \pmb { x } ) - f ( \pmb { x } ^ { \prime } ) ) ^ { 2 } \right] \right) ^ { 1 / 2 } , \qquad \pmb { x } , \pmb { x } ^ { \prime } \in \mathcal { X } .
$$

![](images/c0f6e312d7bc510e954ae871001895989885a8e9d362ee075530311515e0a06f.jpg)  
(a) (d, initial data) = (5, Sphere)

![](images/6daf16a88e09fe5c43000655773a412cfc0e58ed050559e04e8adeeeb374c9fd.jpg)  
(b) (d, initial data) = (10, Sphere)

![](images/9c58d1f2c04c5a7492044836aa30cfba957def40f844a696969000581a08222b.jpg)  
(c) (d, initial data) = (20, Sphere)

![](images/1384b0c570662053b95b6b2ee91042f929d81e9318da37d63ae8706a5ce12a88.jpg)  
(d) (d, initial data)=(5, Rosenbrock)(e)

![](images/4e9dcfe12aeab52c0b458aaa11d7217b5827ebd02ac50b3bd8649c9c114ab775.jpg)  
(d, initial data)  
(10, Rosenbrock)

![](images/98b2cac5ea26bd16620fed1666664edabf90c3c86c7065d852cf3e30a5b2ec70.jpg)  
(20, Rosenbrock)  
(d, initial data)

![](images/2d6d3381e463c4fd07b97d4748f7084d88392723939640db968e0cdb0b0a745d.jpg)  
(g) (d, initial data) = (5, Levy)

![](images/121ad7e69c0939934692f33f66c41c0e75dcf21f716a4e44dcea7924d6456014.jpg)  
(h) (d, initial data) = (10, Levy)

![](images/0fe2ee0a943ee6eacc2e82b989dd9b1caad5caafc7d6a945364394d3a6e2bb83.jpg)  
(i) (d, initial data) = (20, Levy)  
Figure 4: Comparison of test function performance (n<sub>0</sub> = 50).

![](images/5a6655938c10ea41bcdf461c20befc66724b5f51ae892872a3669745b081e726.jpg)  
(a) (d, initial data) = (5, Sphere)

![](images/a3956cfaebeabc222034179b6c7d219a40e467c9e6393022593413282b4aff2b.jpg)  
(b) (d, initial data) = (10, Sphere)

![](images/6f7e787bf6bb24e85cb67f426da58d1af3ad68e882c5b677d4a659a1fcae32a4.jpg)  
(c) (d, initial data) = (20, Sphere)

![](images/6f9575a2466370b871f67ded2e34f326e561721eb1d624a40f6d844e207e89b3.jpg)  
(d) (d, initial data)=(5, Rosenbrock)(e)

![](images/c04f39ba3c14d77bba946de05ca600b7f2b754d7203c0a2e5528ce435b52bfbe.jpg)  
(10, Rosenbrock)  
(d, initial data)

![](images/e62c49d33191ba7dc5cb03ef489be60fe66ea85bb52c9d843e70d334a33fc183.jpg)  
(20, Rosenbrock)  
(d, initial data)

![](images/cb98b46d5319dc8db083d3e038391be014c0fa05db0e5cba04deeee9f3cb594a.jpg)  
(g) (d, initial data) = (5, Levy)

![](images/31fe2ef77e70c4388647800c52ca7fb51487102abf13bfcec427dd0f3b3a650e.jpg)  
(h) (d, initial data) = (10, Levy)

![](images/b3b4c618e3d3bac3a07e8f8011228e7c1c191c79d028fccba0ac1609a3ba4492.jpg)  
(i) (d, initial data) = (20, Levy)  
Figure 5: Comparison of test function performance (n<sub>0</sub> = 300).

![](images/029d5768da5d98df589a79390e48a26107e9037b69b210a373cfed27d419bc83.jpg)  
(a) (d, initial data) = (5d, sph)

![](images/c1c5e300a9a62698a16d4a02ee78d4c797245671622ff122f902d36d4ad9dba2.jpg)  
(b) (d, initial data) = (10d, sph)

![](images/0f87692e147063fa16f7cab0d09fbf13fce5b2ed7b9e3939f7d52807ac9f5bac.jpg)  
(c) (d, initial data) = (20d, sph)

![](images/11f2bdc8b207fcc1ae480441e940cca379fcaec7116d0348732f18cd31dfc21f.jpg)  
(d) (d, initial data) = (5d, ros)

![](images/ec33f03ea39bc185ca8ffcd2009195d308bfda9d2e9cb72a79607be86141b1c5.jpg)  
(e) (d, initial data) = (10d, ros)

![](images/00a2641c40929d7e6409229d930045b5c2b517315485708143cde88bf87bdefc.jpg)  
(f) (d, initial data) = (20d, ros)

![](images/85005232da90e72c12ceb2ebf59adb76db7fe1e755fe153357c41749f9d0e704.jpg)  
(g) (d, initial data) = (5d, lev)

![](images/0d9a016d00054aaec8b222a442d4930054bfae99d847cc25956cabcb3f02f525.jpg)  
(h) (d, initial data) = (10d, lev)

![](images/5f149812b3a5f7fb67ee3f33b296495ac3d41d5c271cf256e1788a6fdbb962f8.jpg)  
(i) (d, initial data) = (20d, lev)  
Figure 6: Comparison of test function performance $( n _ { 0 } = 1 0 0 )$ . Standard error is added to Fig. 1.

Suppose that X is d-compact (compact with respect to the canonical pseudometric d). Let

$$
N ( \mathcal { X } , d , \epsilon )
$$

denote the minimum number of d-balls (balls under the metric d) of radius ϵ required to cover X. Then there exists a universal constant $C > 0$ such that

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { { \boldsymbol x } \in \mathcal { X } } f ( { \boldsymbol x } ) \right] \le C \int _ { 0 } ^ { \mathrm { d i a m } ( \mathcal { X } , d ) } \sqrt { \log N ( \mathcal { X } , d , { \boldsymbol \epsilon } ) } d \boldsymbol \epsilon ,
$$

where

$$
\dim ( { \mathcal { X } } , d ) : = \operatorname* { s u p } _ { x , x ^ { \prime } \in { \mathcal { X } } } d ( x ^ { \prime } , x ^ { \prime } ) .
$$

In particular, if the entropy integral on the right-hand side is finite, then

$$
\mathbb { E } [ \operatorname* { s u p } _ { \pmb { x } \in \mathcal { X } } f ( \pmb { x } ) ] < \infty .
$$

![](images/5c432135c12509191b97f39baa53946468e055c114d5d57e93b53230e49ba41d.jpg)  
(a) (d, initial data) = (5d, sph)

![](images/614b8b65455763ef084864285ef2849657edbb64a5df9ff271861c5283283267.jpg)  
(b) (d, initial data) = (10d, sph)

![](images/04d2d1cfa3dcdeb5382664aa2feca825bf4aef4af720b247832dc8f3c4326369.jpg)  
(c) (d, initial data) = (20d, sph)

![](images/e1134a3e4acbb45992d722899695395439bb632aa6d618b464eb0ee6506068fa.jpg)  
(d) (d, initial data) = (5d, ros)

![](images/2c868339dccdd28e2f11acb135f966dfe1cf76478536fe8461036836a9f6967a.jpg)  
(e) (d, initial data) = (10d, ros)

![](images/27e3fc052cc5803150543d7e41d638891e3723e08a5e7178fed43b4afd413a34.jpg)  
(f) (d, initial data) = (20d, ros)

![](images/66f3ee1841c05810bbe27a8f5a37b364b0c49c4871d4ef720102b197b0319123.jpg)  
(g) (d, initial $\mathrm { d a t a } ) = ( 5 d , \mathrm { l e v } )$

![](images/b3112c6d26a2251efe1b1891244cbead4be612139ab922a436e22ec738aa3d62.jpg)  
(h) (d, initial data) = (10d, lev)

![](images/69ec193dda076638fee23161732b3ef964fe8c3b4c051d85cd11e528de49b710.jpg)  
(i) (d, initial data) = (20d, lev)  
Figure 7: Comparison of test function performance $( n _ { 0 } = 5 0 )$ . Standard error is added to Fig. 4.

Theorem D.2 (Borell–TIS inequality (adapted from Adler and Taylor, 2007)). Let $f = \{ f ( x )$ $x \in \mathcal { X } \}$ be a centered Gaussian process that is almost surely bounded on X. Define

$$
\sigma _ { \mathcal { X } } ^ { 2 } : = \operatorname* { s u p } _ { \pmb { x } \in \mathcal { X } } \mathrm { V a r } ( f ( \pmb { x } ) ) .
$$

Then

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { x \in \mathcal { X } } f ( \pmb { x } ) \right] < \infty ,
$$

and, for every $u > 0$

$$
\mathbb { P } \left( \operatorname* { s u p } _ { \mathbf { x } \in \mathcal { X } } f ( \mathbf { x } ) > \mathbb { E } \left[ \operatorname* { s u p } _ { \mathbf { x } \in \mathcal { X } } f ( \mathbf { x } ) \right] + u \right) \leq \exp \left( - \frac { u ^ { 2 } } { 2 \sigma _ { \mathcal { X } } ^ { 2 } } \right) .
$$

Let

$$
\mathcal { D } _ { 0 } = \{ ( x _ { j } ^ { ( 0 ) } , y _ { j } ^ { ( 0 ) } ) \} _ { j = 1 } ^ { n _ { 0 } }
$$

![](images/67f20ac8eb6c49268917ce08aee7a2068c61c934da0144f00b6cb91ef6fbae20.jpg)  
(a) (d, initial data) = (5d, sph)

![](images/627d92b296914f305deea2c701ff6175301e9d5d95e0c564e8d124d1ac41fbf4.jpg)  
(b) (d, initial data) = (10d, sph)

![](images/e0a3936fe9468094a5d7dd21e765b726dace912af5c67a0d82f9e86770ae04b7.jpg)  
(c) (d, initial data) = (20d, sph)

![](images/3350a91e9e0e7e64325aa63efc52c3b3fb7ff1fd5fe9d40e998ea46cd4d6df27.jpg)  
(d) (d, initial data) = (5d, ros)

![](images/1126d3b63db97327eccdf737f76f596ab2afee754dbeb8969cb81eb3a762e62d.jpg)

![](images/8f597965d4532f8e511d6ef9779717fdbb94af64f43434584585d96dad6d77ec.jpg)  
(f) (d, initial data) = (20d, ros)

(e) (d, initial data) = (10d, ros)  
![](images/adbfb37a352e9bdffe52b66cc68207aeb6113921215288d653486f8a46b994b3.jpg)  
(g) (d, initial $\mathrm { d a t a } ) = ( 5 d , \mathrm { l e v } )$

![](images/26adf650049e7b167755f84b4036cfe2c9704416f2f1ed883a99eea9692ccf8d.jpg)  
(h) (d, initial data) = (10d, lev)

![](images/0cf2b16464f463c33d885f1108ea46c4760140f27303baf86f951d6a0d9eaeec.jpg)  
(i) (d, initial data) = (20d, lev)  
Figure 8: Comparison of test function performance $( n _ { 0 } = 3 0 0 )$ . Standard error is added to Fig. 5.

be a fixed initial dataset and

$$
\begin{array} { r } { g ( x ) : = f ( x ) - m ( x ) , } \\ { g _ { 0 } ( x ) : = f ( x ) - \mu _ { 0 } ( x ) } \end{array}
$$

be the ‘centered’ processes. Then,

$$
\begin{array} { c } { { g \sim \mathcal { G P } ( 0 , k ) , } } \\ { { g _ { 0 } \mid \mathcal { D } _ { 0 } \sim \mathcal { G P } ( 0 , \Sigma _ { 0 } ) . } } \end{array}
$$

We first derive the finiteness of the second moment of g<sub>0</sub> given $\mathcal { D } _ { 0 }$

## Lemma D.1. Let

$$
M _ { 0 } : = \operatorname* { s u p } _ { \pmb { x } \in \mathcal { X } } | g _ { 0 } ( \pmb { x } ) | .
$$

Under the assumptions of Proposition 3.1,

$$
C _ { \mathcal { D } _ { 0 } } = \mathbb { E } [ M _ { 0 } ^ { 2 } \mid \mathcal { D } _ { 0 } ] < \infty .\tag{6}
$$

Proof. From the definition of the canonical pseudo-metric $d ,$

$$
d ( \pmb { x } , \pmb { x } ^ { \prime } ) ^ { 2 } = \mathbb { E } [ \{ g ( \pmb { x } ) - g ( \pmb { x } ^ { \prime } ) \} ^ { 2 } ] = k ( \pmb { x } , \pmb { x } ) + k ( \pmb { x } ^ { \prime } , \pmb { x } ^ { \prime } ) - 2 k ( \pmb { x } , \pmb { x } ^ { \prime } ) .
$$

We further define the canonical pseudo-metric conditioned on $\mathcal { D } _ { 0 }$ as

$$
d _ { 0 } ( \pmb { x } , \pmb { x } ^ { \prime } ) : = \left( \mathbb { E } [ \{ f ( \pmb { x } ) - f ( \pmb { x } ^ { \prime } ) \} ^ { 2 } \ | \ \mathcal { D } _ { 0 } ] \right) ^ { 1 / 2 } .
$$

Then,

$$
d _ { 0 } ( { \pmb x } , { \pmb x } ^ { \prime } ) ^ { 2 } = { \mathbb E } [ \{ g _ { 0 } ( { \pmb x } ) - g _ { 0 } ( { \pmb x } ^ { \prime } ) \} ^ { 2 } \mid \mathcal { D } _ { 0 } ] = \Sigma _ { 0 } ( { \pmb x } , { \pmb x } ) + \Sigma _ { 0 } ( { \pmb x } ^ { \prime } , { \pmb x } ^ { \prime } ) - 2 \Sigma _ { 0 } ( { \pmb x } , { \pmb x } ^ { \prime } )\tag{7}
$$

Let $k _ { X _ { 0 } } ( \pmb { x } ) : = [ k ( \pmb { x } _ { 1 } ^ { ( 0 ) } , \pmb { x } ) , \dots , k ( \pmb { x } _ { n _ { 0 } } ^ { ( 0 ) } , \pmb { x } ) ] ^ { \top }$ and $v ( x , x ^ { \prime } ) : = k _ { X _ { 0 } } ( \pmb { x } ) - k _ { X _ { 0 } } ( \pmb { x } ^ { \prime } )$ . By expanding the GP posterior covariance $\Sigma _ { 0 }$ in (7), we have

$$
\begin{array} { l } { { d _ { 0 } ( { \pmb x } , { \pmb x } ^ { \prime } ) ^ { 2 } = k ( { \pmb x } , { \pmb x } ) + \sigma _ { \varepsilon } ^ { 2 } - k _ { X _ { 0 } } ( { \pmb x } ) ^ { \top } ( K _ { 0 } + \sigma _ { \varepsilon } ^ { 2 } I ) ^ { - 1 } k _ { X _ { 0 } } ( { \pmb x } ) } } \\ { { \qquad + k ( { \pmb x } ^ { \prime } , { \pmb x } ^ { \prime } ) + \sigma _ { \varepsilon } ^ { 2 } - k _ { X _ { 0 } } ( { \pmb x } ^ { \prime } ) ^ { \top } ( K _ { 0 } + \sigma _ { \varepsilon } ^ { 2 } I ) ^ { - 1 } k _ { X _ { 0 } } ( { \pmb x } ^ { \prime } ) } } \\ { { \qquad - 2 \left( k ( { \pmb x } , { \pmb x } ^ { \prime } ) + \sigma _ { \varepsilon } ^ { 2 } - k _ { X _ { 0 } } ( { \pmb x } ) ^ { \top } ( K _ { 0 } + \sigma _ { \varepsilon } ^ { 2 } I ) ^ { - 1 } k _ { X _ { 0 } } ( { \pmb x } ^ { \prime } ) \right) } } \\ { { \qquad = d ( { \pmb x } , { \pmb x } ^ { \prime } ) - v ( { \pmb x } , { \pmb x } ^ { \prime } ) ( K _ { 0 } + \sigma _ { \varepsilon } ^ { 2 } I ) ^ { - 1 } v ( { \pmb x } , { \pmb x } ^ { \prime } ) . } } \end{array}
$$

Since $K _ { 0 } + \sigma _ { \varepsilon } ^ { 2 } I \succ 0$ , the second term is positive, by which we see

$$
d _ { 0 } ( { \pmb x } , { \pmb x } ^ { \prime } ) \leq d ( { \pmb x } , { \pmb x } ^ { \prime } ) \qquad \forall { \pmb x } , { \pmb x } ^ { \prime } \in \mathcal { X } .
$$

As a result, from the assumption (4), there exists $r _ { 0 }$ that satisfies

$$
d _ { 0 } ( \pmb { x } , \pmb { x } ^ { \prime } ) \leq L \| \pmb { x } - \pmb { x } ^ { \prime } \| _ { 2 } ^ { \alpha } \qquad \mathrm { i f } \ \| \pmb { x } - \pmb { x } ^ { \prime } \| _ { 2 } \leq r _ { 0 } .\tag{8}
$$

For suficiently small $\epsilon > 0$ , define

$$
r _ { \epsilon } : = \left( \frac { \epsilon } { L } \right) ^ { 1 / \alpha } .\tag{9}
$$

In particular, if $0 < \epsilon \leq L r _ { 0 } ^ { \alpha }$ , then $r _ { \epsilon } \leq r _ { 0 }$ . Whenever $\| \pmb { x } - \pmb { x } ^ { \prime } \| _ { 2 } \leq r _ { \epsilon }$

$$
d _ { 0 } ( \pmb { x } , \pmb { x } ^ { \prime } ) \leq L \| \pmb { x } - \pmb { x } ^ { \prime } \| _ { 2 } ^ { \alpha } \leq L r _ { \epsilon } ^ { \alpha } = \epsilon .
$$

Therefore, every Euclidean r<sub>ϵ</sub>-cover of X is also a d<sub>0</sub>-ϵ-cover, and thus

$$
N ( \mathcal { X } , d _ { 0 } , \epsilon ) \leq N ( \mathcal { X } , \| \cdot \| _ { 2 } , r _ { \epsilon } ) ,\tag{10}
$$

where

$$
N ( \mathcal { X } , \rho , \epsilon )
$$

denote the minimum number of balls of radius ϵ, defined by a metric $\rho ,$ that covers $\mathcal { X } .$

Since $\mathcal { X } \subset \mathbb { R } ^ { d }$ is compact, it is contained in a Euclidean ball $B _ { 2 } ^ { d } ( R )$ of some finite radius R. The standard volumetric covering-number bound for Euclidean balls (Vershynin, 2018) gives $\begin{array} { r } { N ( B _ { 2 } ^ { d } ( R ) , \parallel \cdot \parallel _ { 2 } , r ) \le \left( 1 + \frac { 2 R } { r } \right) ^ { d } } \end{array}$ . For suficiently small $r > 0$

$$
N ( \mathcal { X } , \| \cdot \| _ { 2 } , r ) \leq N ( B _ { 2 } ^ { d } ( R ) , \| \cdot \| _ { 2 } , r ) \leq \left( \frac { 3 R } { r } \right) ^ { d } = A { _ { \mathcal { X } } } r ^ { - d } ,\tag{11}
$$

where $A _ { { \mathcal { X } } } : = ( 3 R ) ^ { d } < \infty$ . Combining (10), (11) and (9), for all suficiently small $\epsilon > 0$

$$
\begin{array} { l } { { N ( \mathcal { X } , d _ { 0 } , \epsilon ) \le A _ { \mathcal { X } } r _ { \epsilon } ^ { - d } } } \\ { { \ } } \\ { { \displaystyle = A _ { \mathcal { X } } \left( \frac { L } { \epsilon } \right) ^ { d / \alpha } . } } \end{array}
$$

This indicates that $( \mathcal { X } , d _ { 0 } )$ is totally bounded $( N ( \mathcal { X } , d _ { 0 } , \epsilon ) < \infty$ for $\forall \epsilon > 0 )$ (Wu, 2020). Then,

$$
\log N ( \mathcal { X } , d _ { 0 } , \epsilon ) \leq \log A _ { \mathcal { X } } + \frac { d } { \alpha } \log \left( \frac { L } { \epsilon } \right)\tag{12}
$$

for all suficiently small $\epsilon > 0$

Define

$$
S _ { 0 } ^ { + } : = \operatorname* { s u p } _ { { \pmb x } \in \mathcal { X } } g _ { 0 } ( { \pmb x } ) , \qquad S _ { 0 } ^ { - } : = \operatorname* { s u p } _ { { \pmb x } \in \mathcal { X } } \{ - g _ { 0 } ( { \pmb x } ) \} .
$$

Since $g _ { 0 }$ is the centered posterior, we see $S _ { 0 } ^ { + } \overset { d } { = } S _ { 0 } ^ { - }$ given $\mathcal { D } _ { 0 }$ . Let

$$
m _ { 0 } : = \mathbb { E } \left[ \left. S _ { 0 } ^ { + } \right| \mathcal { D } _ { 0 } \right] = \mathbb { E } \left[ \left. S _ { 0 } ^ { - } \right| \mathcal { D } _ { 0 } \right] .
$$

According to Dudley’s entropy theorem D.1, for totally bounded $( \mathcal { X } , d _ { 0 } )$ , we see

$$
m _ { 0 } \le C J _ { 0 }
$$

where

$$
J _ { 0 } : = \int _ { 0 } ^ { \mathrm { d i a m } ( \mathcal { X } , d _ { 0 } ) } \sqrt { \log N ( \mathcal { X } , d _ { 0 } , { \epsilon } ) } d \epsilon
$$

$C < \infty$ is a constant, and diam $\begin{array} { r } { . ( { \mathcal X } , d _ { 0 } ) : = \operatorname* { s u p } _ { x , x ^ { \prime } \in { \mathcal X } } d _ { 0 } ( x , x ^ { \prime } ) } \end{array}$ , which is finite because of $d _ { 0 } ( { \pmb x } , { \pmb x } ^ { \prime } ) \leq$ $L \| \mathbf { \boldsymbol { x } } - \mathbf { \boldsymbol { x } } ^ { \prime } \| _ { 2 } ^ { \alpha }$ . The finiteness of $J _ { 0 }$ is confirmed through (12). From (12), we have $\sqrt { \log N ( \mathcal { X } , d _ { 0 } , \epsilon ) } =$ $O \left( \sqrt { \log ( 1 / \epsilon ) } \right)$ when $\epsilon \downarrow 0 .$ . There exists $\epsilon _ { 0 } > 0$ by which any $\epsilon \leq \epsilon _ { 0 }$ satisfies (12).

$$
\begin{array} { c l } { \displaystyle { J _ { 0 } = \int _ { 0 } ^ { \epsilon _ { 0 } } \sqrt { \log { N ( \mathcal { X } , d _ { 0 } , \epsilon ) } } d \epsilon + \int _ { \epsilon _ { 0 } } ^ { \dim ( \mathcal { X } , d _ { 0 } ) } \sqrt { \log { N ( \mathcal { X } , d _ { 0 } , \epsilon ) } } d \epsilon } } \\ { \displaystyle { \le \int _ { 0 } ^ { \epsilon _ { 0 } } \sqrt { \log { N ( \mathcal { X } , d _ { 0 } , \epsilon ) } } d \epsilon + \underbrace { ( \dim ( \mathcal { X } , d _ { 0 } ) - \epsilon _ { 0 } ) \sqrt { \log { N ( \mathcal { X } , d _ { 0 } , \epsilon _ { 0 } ) } } } _ { < \infty } } } \end{array}
$$

The inequality of the second line is from the non-increasing property of the covering number with respect to ϵ. Since $\sqrt { \log N ( \mathcal { X } , d _ { 0 } , \epsilon ) } = O ( \sqrt { \log ( 1 / \epsilon ) } )$ for $\epsilon \leq \epsilon _ { 0 }$ , we only need to show the finiteness $\textstyle \int _ { 0 } ^ { \epsilon _ { 0 } } { \sqrt { \log ( 1 / \epsilon ) } } d \epsilon$ . This can be seen from $\begin{array} { r } { \int _ { 0 } ^ { \epsilon _ { 0 } } \sqrt { \log ( 1 / \epsilon ) } d \epsilon = \int _ { \log ( 1 / \epsilon _ { 0 } ) } ^ { \infty } \sqrt { u } e ^ { - u } } \end{array}$ du < ∞. As a result, we see

$$
m _ { 0 } < \infty .
$$

Since $m _ { 0 } < \infty ,$ we see $S _ { 0 } ^ { + } < \infty$ and $S _ { 0 } ^ { - } ~ < ~ \infty$ almost surely (cannot be $\infty$ with a positive probability) for the given $\mathcal { D } _ { 0 }$ . This means that g<sub>0</sub> conditioned on $\mathcal { D } _ { 0 }$ is almost surely bounded.

Since $g _ { 0 }$ given $\mathcal { D } _ { 0 }$ is almost surely bounded, we can apply Borell–TIS theorem D.2 to $g _ { 0 }$ and to $- { g _ { 0 } }$ , and obtain

$$
\mathbb { P } \left( S _ { 0 } ^ { + } > m _ { 0 } + u \middle | \mathcal { D } _ { 0 } \right) \leq \exp \left( - \frac { u ^ { 2 } } { 2 \sigma _ { 0 } ^ { 2 } } \right) , \qquad \mathrm { f o r ~ } u \geq 0 .\tag{13}
$$

$$
\mathbb { P } \left( S _ { 0 } ^ { - } > m _ { 0 } + u \big | \mathcal { D } _ { 0 } \right) \le \exp \left( - \frac { u ^ { 2 } } { 2 \sigma _ { 0 } ^ { 2 } } \right) , \qquad \mathrm { ~ f o r ~ } u \ge 0 ,\tag{14}
$$

where

$$
\sigma _ { 0 } ^ { 2 } : = \operatorname* { s u p } _ { \pmb { x } \in \mathcal { X } } \Sigma _ { 0 } ( \pmb { x } , \pmb { x } ) .
$$

Note that $\sigma _ { 0 }$ is finite because we have $0 \leq \Sigma _ { 0 } ( x , x ) \leq k ( x , x )$ from the posterior covariance formula, and $\begin{array} { r } { \operatorname* { s u p } _ { x \in \mathcal { X } } k ( \pmb { x } , \pmb { x } ) < 0 } \end{array}$ ∞ since k is continuous and $\mathcal { X }$ is compact.

Let

$$
M _ { 0 } : = \operatorname* { s u p } _ { \pmb { x } \in \mathcal { X } } \vert g _ { 0 } ( \pmb { x } ) \vert = \operatorname* { m a x } \{ S _ { 0 } ^ { + } , S _ { 0 } ^ { - } \} .
$$

For every $u \geq 0$

$$
\left\{ M _ { 0 } > m _ { 0 } + u \right\} = \left\{ S _ { 0 } ^ { + } > m _ { 0 } + u \right\} \cup \left\{ S _ { 0 } ^ { - } > m _ { 0 } + u \right\} .
$$

By the union bound together with (13) and (14), we see

$$
\mathbb { P } \left( M _ { 0 } > m _ { 0 } + u | \mathcal { D } _ { 0 } \right) \leq 2 \exp \left( - \frac { u ^ { 2 } } { 2 \sigma _ { 0 } ^ { 2 } } \right) , \qquad u \geq 0 .\tag{15}
$$

We now use this Gaussian tail bound to establish the finiteness of the second moment of $M _ { 0 }$ For any nonnegative random variable $Z ,$

$$
\mathbb { E } [ Z ^ { 2 } ] = 2 \int _ { 0 } ^ { \infty } t \mathbb { P } ( Z > t ) d t .
$$

Applying this identity conditionally on $\mathcal { D } _ { 0 }$ to $M _ { 0 }$ gives

$$
{ \mathbb E } [ M _ { 0 } ^ { 2 } \mid { \mathcal D } _ { 0 } ] = 2 \int _ { 0 } ^ { \infty } t { \mathbb P } ( M _ { 0 } > t \mid { \mathcal D } _ { 0 } ) d t .
$$

Split the integral at $m _ { 0 } \colon$

$$
\begin{array} { r l r } {  { \mathbb { E } [ M _ { 0 } ^ { 2 } \mid \mathcal { D } _ { 0 } ] = 2 \int _ { 0 } ^ { m _ { 0 } } t \mathbb { P } \big ( M _ { 0 } > t \mid \mathcal { D } _ { 0 } \big ) d t } } \\ & { } & { + \ 2 \int _ { m _ { 0 } } ^ { \infty } t \mathbb { P } ( M _ { 0 } > t \mid \mathcal { D } _ { 0 } ) d t . } \end{array}
$$

The first term is bounded by

$$
2 \int _ { 0 } ^ { m _ { 0 } } t d t = ( m _ { 0 } ) ^ { 2 } .
$$

For the second term, write $t = m _ { 0 } + u$ and use (15):

$$
\begin{array} { r l r } {  { 2 \int _ { m _ { 0 } } ^ { \infty } t \mathbb { P } ( M _ { 0 } > t \mid { \mathcal D } _ { 0 } ) d t } } \\ & { } & { \leq 4 \int _ { 0 } ^ { \infty } ( m _ { 0 } + u ) \exp ( - \frac { u ^ { 2 } } { 2 \sigma _ { 0 } ^ { 2 } } ) d u . } \end{array}
$$

Both integrals on the right-hand side are finite, since

$$
\int _ { 0 } ^ { \infty } \exp \left( - \frac { u ^ { 2 } } { 2 \sigma _ { 0 } ^ { 2 } } \right) d u < \infty
$$

and

$$
\int _ { 0 } ^ { \infty } u \exp \left( - \frac { u ^ { 2 } } { 2 \sigma _ { 0 } ^ { 2 } } \right) d u = \sigma _ { 0 } ^ { 2 } < \infty .
$$

Therefore

$$
\mathbb { E } [ M _ { 0 } ^ { 2 } \mid { \mathcal { D } } _ { 0 } ] < \infty .
$$

Based on lemma D.1, we derive the proof for Proposition 3.1

Proof. From the definition of $g _ { 0 }$

$$
f ( { \pmb x } ) = \mu _ { 0 } ( { \pmb x } ) + g _ { 0 } ( { \pmb x } ) .\tag{16}
$$

Under our assumptions, the posterior mean function is continuous on the compact set $\mathcal { X } .$ Consequently,

$$
\| \mu _ { 0 } \| _ { \infty } : = \operatorname* { s u p } _ { x \in \mathcal { X } } | \mu _ { 0 } ( \pmb { x } ) | < \infty .
$$

By (16),

$$
\operatorname* { s u p } _ { \pmb { x } \in \mathcal { X } } | f ( \pmb { x } ) | \leq \| \mu _ { 0 } \| _ { \infty } + M _ { 0 } .
$$

Therefore, from $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$

$$
\left( \operatorname* { s u p } _ { \mathbf { x } \in \mathcal { X } } \left| f ( \mathbf { x } ) \right| \right) ^ { 2 } \leq 2 \| \mu _ { 0 } \| _ { \infty } ^ { 2 } + 2 M _ { 0 } ^ { 2 } .
$$

Taking conditional expectations and using (6) gives

$$
\begin{array} { r l } & { C _ { \mathcal { D } _ { 0 } } : = \mathbb { E } \left[ \left. \left( \underset { x \in \mathcal { X } } { \operatorname* { s u p } } \left| f ( x ) \right| \right) ^ { 2 } \right| \mathcal { D } _ { 0 } \right] } \\ & { \qquad \leq 2 \| \mu _ { 0 } \| _ { \infty } ^ { 2 } + 2 \mathbb { E } [ M _ { 0 } ^ { 2 } \mid \mathcal { D } _ { 0 } ] } \\ & { \qquad < \infty . } \end{array}
$$

For every realization of the posterior sample path, future observation noises, and algorithmic randomization,

$$
\begin{array} { r l r } & { 0 \leq f ^ { \star } - f ( { \pmb x } _ { t } ) \leq | f ^ { \star } | + | f ( { \pmb x } _ { t } ) | } & \\ & { } & { \leq 2 \underset { { \pmb x } \in \mathcal { X } } { \operatorname* { s u p } } | f ( { \pmb x } ) | . } \end{array}
$$

Note that this inequality is pathwise and is independent of how $\mathbf { \Delta } \mathbf { x } _ { t }$ was selected. Summing over $t = 1 , \dots , T$ yields

$$
0 \leq R _ { T } \leq 2 T \operatorname* { s u p } _ { { \pmb x } \in { \pmb X } } | f ( { \pmb x } ) | .
$$

Hence

$$
R _ { T } ^ { 2 } \leq 4 T ^ { 2 } \left( \operatorname* { s u p } _ { { \pmb x } \in { \pmb X } } | f ( { \pmb x } ) | \right) ^ { 2 } .
$$

Taking conditional expectations and using (6),

$$
\mathbb { E } [ R _ { T } ^ { 2 } \mid { \mathcal { D } } _ { 0 } ] \leq 4 C _ { { \mathcal { D } } _ { 0 } } T ^ { 2 } .
$$

Therefore, we see

$$
\operatorname { V a r } ( R _ { T } \mid { \mathcal { D } } _ { 0 } ) \leq 4 C _ { { \mathcal { D } } _ { 0 } } T ^ { 2 } .
$$

Since the MC estimator $\widehat { \mathrm { B R } } _ { T }$ is unbiased and each $R _ { T }$ is independent,

$$
\begin{array} { r l } & { \mathbb { E } [  ( \widehat { \mathrm { B R } _ { T } } ( \pmb { \theta } ) - \mathrm { B R } _ { T } ( \pmb { \theta } ) ) ^ { 2 } | \mathcal { D } _ { 0 } ] } \\ & { \qquad = \mathrm { V a r } ( \widehat { \mathrm { B R } _ { T } } ( \pmb { \theta } ) | \mathcal { D } _ { 0 } ) } \\ & { \qquad = \displaystyle \frac { 1 } { N } \mathrm { V a r } ( R _ { T } \mid \mathcal { D } _ { 0 } ) } \\ & { \qquad \leq \frac { 4 C _ { \mathcal { D } _ { 0 } } T ^ { 2 } } { N } . } \end{array}\tag{17}
$$

(Conditional) Chebyshev’s inequality and (17) derive

$$
\mathbb { P } \bigg ( \bigg | \widehat { \mathrm { B R } } _ { T } ( \pmb { \theta } ) - \mathrm { B R } _ { T } ( \pmb { \theta } ) \bigg | > \eta \bigg | \mathcal { D } _ { 0 } \bigg ) \leq \frac { 4 C _ { \mathcal { D } _ { 0 } } T ^ { 2 } } { N \eta ^ { 2 } } .
$$

Taking $\eta = 2 T \sqrt { \frac { C _ { \mathcal { D } _ { 0 } } } { N \delta } }$ gives

$$
\mathbb { P } \left( \left| { \widehat { \mathrm { B R } } } _ { T } ( \pmb { \theta } ) - { \mathrm { B R } } _ { T } ( \pmb { \theta } ) \right| \leq 2 T \sqrt { \frac { C _ { { \mathscr { D } } _ { 0 } } } { N \delta } } \Bigg | \mathscr { D } _ { 0 } \right) \geq 1 - \delta
$$

for $\delta \in ( 0 , 1 )$ . Therefore, we obtain

$$
\widehat { \mathrm { B R } } _ { T } ( \pmb \theta ) - \mathrm { B R } _ { T } ( \pmb \theta ) = O _ { p } \left( \frac { T } { \sqrt { N } } \right)
$$

conditionally on $\mathcal { D } _ { 0 }$

## E Bounds of Canonical Pseudo-Metric

Suppose first that k is the Gaussian kernel

$$
k ( \pmb { x } , \pmb { x } ^ { \prime } ) = \sigma _ { f } ^ { 2 } \exp \left( - \frac { \| \pmb { x } - \pmb { x } ^ { \prime } \| _ { 2 } ^ { 2 } } { 2 \ell ^ { 2 } } \right) ,
$$

where $\ell > 0$ is the length-scale parameter and $\sigma _ { f } > 0$ is the scale parameter. Writing $r = \| \pmb { x } - \pmb { x } ^ { \prime } \| _ { 2 }$ we obtain

$$
d ( { \pmb x } , { \pmb x } ^ { \prime } ) ^ { 2 } = 2 \sigma _ { f } ^ { 2 } \left( 1 - \exp \left\{ - \frac { r ^ { 2 } } { 2 \ell ^ { 2 } } \right\} \right)
$$

$$
\leq \frac { \sigma _ { f } ^ { 2 } } { \ell ^ { 2 } } r ^ { 2 } ,
$$

where $1 - e ^ { - a } \leq a$ for $a \geq 0$ was used. Hence

$$
d ( \pmb { x } , \pmb { x } ^ { \prime } ) \leq \frac { \sigma _ { f } } { \ell } \| \pmb { x } - \pmb { x } ^ { \prime } \| _ { 2 } .
$$

We define the Mat´ern kernel by

$$
k _ { \nu } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \sigma _ { f } ^ { 2 } \frac { 2 ^ { 1 - \nu } } { \Gamma ( \nu ) } \left( \frac { \sqrt { 2 \nu } } { \ell } \lVert \mathbf { x } - \mathbf { x } ^ { \prime } \rVert _ { 2 } \right) ^ { \nu } K _ { \nu } \left( \frac { \sqrt { 2 \nu } } { \ell } \lVert \mathbf { x } - \mathbf { x } ^ { \prime } \rVert _ { 2 } \right) ,
$$

where $\nu > 0$ is the smoothness parameter, $\ell > 0$ is the length-scale parameter, $\sigma _ { f } > 0$ is the scale parameter, $\Gamma ( \cdot )$ denotes the Gamma function, and $K _ { \nu } ( \cdot )$ denotes the modified Bessel function of the second kind of order ν. For the canonical pseudo-metric of a Mat´ern kernel, it is known that

$$
\gamma ( h ) = \frac { 1 } { 2 } \mathbb { E } [ \{ f ( x + h ) - f ( x ) \} ^ { 2 } ] = \left\{ \begin{array} { l l } { O ( \| h \| ^ { 2 \nu } ) , } & { 0 < \nu < 1 , } \\ { O ( \| h \| ^ { 2 } \log ( 1 / \| h \| ) ) , } & { \nu = 1 , } \\ { O ( \| h \| ^ { 2 } ) , } & { \nu > 1 , } \end{array} \right. \qquad \| h \| \downarrow 0 .
$$

See (Stein, 1999) for detail. Let $r = \| h \| _ { 2 }$ . The case of $\nu = 1$ is bounded by

$$
O ( r ^ { 2 } | \log ( 1 / r ) | )
$$

Since any positive power dominates a logarithmic factor as $r \downarrow 0$

$$
r ^ { 1 - \alpha } \sqrt { | \log r | } \to 0
$$

for any $\alpha \in ( 0 , 1 )$ and $r \downarrow 0$ . This indicates for suficiently small r

$$
r { \sqrt { | \log r | } } \leq r ^ { \alpha } .
$$

Since

$$
d ( { \pmb x } , { \pmb x } ^ { \prime } ) = \sqrt { 2 \gamma ( { \pmb x } - { \pmb x } ^ { \prime } ) } ,
$$

it follows that, for suficiently small $\| { \pmb x } - { \pmb x } ^ { \prime } \| _ { 2 }$ and some $L < \infty$ 2

$$
d ( \pmb { x } , \pmb { x } ^ { \prime } ) \leq L \| \pmb { x } - \pmb { x } ^ { \prime } \| _ { 2 } ^ { \alpha } ,
$$

where $\alpha = \nu$ for $0 < \nu < 1$ , any $\alpha \in ( 0 , 1 )$ for $\nu = 1$ , and $\alpha = 1$ for $\nu > 1$

## F Proof of Corollary 3.1

Define

$$
\Delta _ { N } : = \operatorname* { m a x } _ { \pmb { \theta } \in \Theta } \left| \widehat { \mathrm { B R } } _ { T } ( \pmb { \theta } ) - \mathrm { B R } _ { T } ( \pmb { \theta } ) \right| .
$$

Since Θ is assumed to be a finite set, we can write $\boldsymbol { \Theta } = \{ \pmb { \theta } _ { 1 } , \dots , \pmb { \theta } _ { K } \}$ for the number of candidate configurations $K > 0$ . For any $\varepsilon > 0$ and each $k = 1 , \ldots , K$ , Proposition 3.1 implies that there exist constants $M _ { k } < \infty$ and $N _ { k }$ such that, for all $N \geq N _ { k }$

$$
\mathrm { P r } \bigg ( \Big | \widehat { \mathrm { B R } } _ { T , N } ( \pmb \theta _ { k } ) - \mathrm { B R } _ { T } ( \pmb \theta _ { k } ) \Big | > M _ { k } \frac { T } { \sqrt { N } } \bigg ) < \frac { \varepsilon } { K } .
$$

Let $M = \operatorname* { m a x } _ { 1 \leq k \leq K } M _ { k }$ and $N _ { 0 } = \mathrm { m a x } _ { 1 \leq k \leq K } N _ { k }$ . Then, by the union bound, for all $N \geq N _ { 0 }$

$$
\begin{array} { r l } & { \operatorname* { P r } \biggr ( \Delta _ { N } > M \displaystyle \frac { T } { \sqrt { N } } \biggr ) \leq \displaystyle \sum _ { k = 1 } ^ { K } \operatorname* { P r } \biggr ( \Bigl | \widehat { \mathrm { B R } } _ { T , N } \bigl ( \pmb { \theta } _ { k } \bigr ) - \mathrm { B R } _ { T } \bigl ( \pmb { \theta } _ { k } \bigr ) \Bigr | > M \displaystyle \frac { T } { \sqrt { N } } \biggr ) } \\ & { \qquad < \varepsilon . } \end{array}
$$

Therefore, we obtain $\Delta _ { N } = O _ { p } ( T / \sqrt { N } )$

By the definition of $\theta _ { N }$

$$
\widehat { \mathrm { B R } } _ { T } ( \widehat { \pmb { \theta } } _ { N } ) \leq \widehat { \mathrm { B R } } _ { T } ( \pmb { \theta } ^ { \star } ) .
$$

Therefore,

$$
\begin{array} { r l } & { \mathrm { B R } _ { T } ( \widehat { \pmb \theta } _ { N } ) - \mathrm { B R } _ { T } ( \pmb \theta ^ { \star } ) = \left[ \mathrm { B R } _ { T } ( \widehat { \pmb \theta } _ { N } ) - \widehat { \mathrm { B R } } _ { T , N } ( \widehat { \pmb \theta } _ { N } ) \right] } \\ & { \quad \quad \quad \quad + \left[ \widehat { \mathrm { B R } } _ { T , N } ( \widehat { \pmb \theta } _ { N } ) - \widehat { \mathrm { B R } } _ { T , N } ( \pmb \theta ^ { \star } ) \right] } \\ & { \quad \quad \quad \quad + \left[ \widehat { \mathrm { B R } } _ { T , N } ( \pmb \theta ^ { \star } ) - \mathrm { B R } _ { T } ( \pmb \theta ^ { \star } ) \right] } \\ & { \quad \quad \quad \quad \leq \left| \mathrm { B R } _ { T } ( \widehat { \pmb \theta } _ { N } ) - \widehat { \mathrm { B R } } _ { T , N } ( \widehat { \pmb \theta } _ { N } ) \right| + \left| \widehat { \mathrm { B R } } _ { T , N } ( \pmb \theta ^ { \star } ) - \mathrm { B R } _ { T } ( \pmb \theta ^ { \star } ) \right| } \\ & { \quad \quad \quad \quad \leq 2 \Delta _ { N } . } \end{array}
$$

Since $\Delta _ { N } = O _ { p } ( T / \sqrt { N } )$ , it follows that

$$
\mathrm { B R } _ { T } ( \widehat { \pmb { \theta } } _ { N } ) - \mathrm { B R } _ { T } ( \pmb { \theta } ^ { \star } ) = O _ { p } \bigg ( \frac { T } { \sqrt { N } } \bigg ) ,
$$

which proves the result.