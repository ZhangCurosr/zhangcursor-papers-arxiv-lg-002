# POD-CNF: reduced order modeling of parametrized stochastic PDEs via normalizing flows and proper orthogonal decomposition

Andrea Violante<sup>1</sup>, Nicola Rares Franco<sup>1</sup>, Andrea Manzoni<sup>1</sup>

<sup>1</sup>MOX, Department of Mathematics, Politecnico di Milano, Piazza Leonardo da Vinci 32, 20133, Milan, Italy

## Abstract

Reduced-order models (ROMs) learned from trusted numerical simulations enable the real-time simulation of complex physical systems, making otherwise computationally prohibitive tasks—such as uncertainty quantification and optimal control—tractable. In particular, data-driven ROMs are opening new opportunities for developing powerful surrogate models that can learn from both numerical simulations and experimental data. However, integrating these sources of information is far from straightforward. Real-world measurements often involve multiple sources of uncertainty, including non-Gaussian noise, latent heterogeneities, unknown or hidden system parameters, and complex multiscale phenomena occurring below the resolution of the available data. Consequently, real-world data are more naturally described by parametrized stochastic partial diferential equations (SPDEs), for which the development of efective ROMs remains relatively unexplored. In this setting, the system response is better characterized by a probability distribution than by a deterministic field, motivating the development of probabilistic surrogates. In this work, we introduce POD-CNF, a data-driven generative ROM for eficient uncertainty quantification and characterization of parametrized high-dimensional systems. The proposed approach combines projection-based model reduction with deep generative modeling. First, Proper Orthogonal Decomposition (POD) is employed to obtain a low-dimensional representation of the solution manifold. A Conditional Normalizing Flow (CNF) is then trained to approximate the conditional probability density of the reduced coeficients given the input parameters. This formulation allows the surrogate to capture complex and potentially multimodal output distributions, going beyond standard regression-based ROMs that typically focus on mean least-squares predictions. We assess the efectiveness of POD-CNF through a series of numerical experiments, demonstrating its ability to accurately reproduce output distributions while substantially reducing computational costs. Furthermore, a proof-of-concept application illustrates how the generative point of view provided by the normalizing flow facilitates the integration of POD-CNF into Bayesian inverse problems, highlighting promising directions for future research.

Keywords: reduced order modeling, scientific machine learning, stochastic & parametrized PDEs, generative mod eling, conditional normalizing flows.

## 1 Introduction

Partial Diferential Equations (PDEs) constitute a standard mathematical framework for modeling physical phenomena in engineering and applied sciences [47]. Given suitable boundary and initial conditions, their solutions can be approx imated numerically through high-fidelity discretization techniques such as the Finite Element Method (FEM), Finite Volume Method (FVM), or spectral methods [47]. In many practical applications, however, the governing equations depend on a set of input parameters describing material properties, geometric configurations, operating conditions, or boundary data. Consequently, one is not interested in solving a single PDE instance, but rather an entire family of parametrized problems. From an abstract point of view, these latter can be expressed, in the stationary case, as the solution of a parametrized PDE of the form

$$
\mathcal { L } ( u , \mu ) = f ( \pmb { \mu } ) \qquad \mathrm { i n } \ \Omega ,\tag{1}
$$

complemented by suitable boundary conditions, where $\Omega \subset \mathbb { R } ^ { d }$ denotes a bounded spatial domain and $\pmb { \mu } \in \mathbb { R } ^ { p }$ a vector of deterministic parameters. Here, L and $f$ denote a parametrized diferential operator and source term, respectively. Under suitable assumptions, the problem is well posed, meaning that for every parameter value $\pmb { \mu }$ there exists a unique solution field $u = u ( \cdot , \mu )$

While high-fidelity numerical solvers provide accurate approximations of (1), repeatedly solving a diferential problem for many parameter configurations may become computationally prohibitive. This issue arises in several many query scenarios, including uncertainty quantification, optimization, inverse problems, and real-time control. Such challenges have motivated the development of Reduced Order Modeling (ROM) techniques, whose objective is to con struct low-dimensional surrogate models capable of rapidly approximating the solution manifold associated with a given parametrized PDE [40, 48, 27, 19].

Among ROM methodologies, projection-based techniques such as Proper Orthogonal Decomposition (POD) have proven particularly efective in extracting compact low-dimensional representations from high-fidelity simulations. More recently, advances in machine learning and the increasing availability of data have fostered the emergence of datadriven reduced-order models, where neural networks and statistical learning techniques are employed to approximate reduced dynamics or parameter-to-solution maps directly from data. These approaches are progressively opening the possibility of constructing surrogate models capable of learning not only from numerical simulations but also from experimental observations. However, the integration of experimental data into ROM frameworks is far from being straightforward. Classical reduced-order models are typically formulated under the assumption that the system response is uniquely determined by a finite-dimensional parameter vector. While this assumption is often appropriate in simulation-driven settings, it becomes restrictive when dealing with real-world measurements. In fact, experimental observations frequently contain variability that cannot be fully explained by the observable parameters alone. Such variability may arise from, e.g., measurement noise, unresolved multiscale phenomena, latent heterogeneities, unknown external forcings, or partially observed operating conditions, to mention a few relevant examples. As a consequence, two experiments performed under nominally identical observable conditions may yield significantly diferent system responses.

In these settings, the deterministic parameter-to-solution paradigm underlying standard ROM methodologies becomes inadequate. Rather than learning a deterministic map $\mu \mapsto u ( \cdot , \pmb { \mu } )$ , one should instead aim at characterizing a conditional probability distribution over possible system responses. $\mathrm { A }$ natural mathematical framework for formalizing this uncertainty is provided by parametrized stochastic PDEs (SPDEs). Let $( \mathcal { E } , \mathcal { F } , \mathbb { P } )$ be a probability space, and let $\boldsymbol \omega \in \mathcal E$ represent a generic event in such space. The nonobservability of such events serves to model our partial knowledge of the system, thus accounting for, $\mathrm { e . g . }$ , unknown sources of variability or unresolved scales. We then consider governing equations of the form

$$
\tilde { \mathcal { L } } ( \boldsymbol { u } , \boldsymbol { \mu } , \omega ) = \tilde { f } ( \boldsymbol { \mu } , \omega ) \qquad \mathrm { i n } ~ \Omega ,\tag{2}
$$

where $\pmb { \mu }$ represents observable deterministic parameters, while ω models hidden stochastic efects. The solution field $u = u ( \pmb { x } , \pmb { \mu } , \omega )$ is therefore no longer deterministic, but rather described by a random field conditioned on the observable parameters $\pmb { \mu } .$ The distinction between (1) and (2) is conceptually important. Deterministic parametrized PDEs can often be interpreted as simplified models of more complex stochastic physical phenomena. For example, in porous media flow, a deterministic nonlinear Darcy model of the form $- \nabla \cdot ( u ^ { \mu - 1 } \nabla u ) = f$ may represent an idealized approximation, whereas a more realistic formulation accounting for unresolved material heterogeneities could read as $- \nabla \cdot ( a ( { \pmb x } , \omega ) u ^ { \mu - 1 } \nabla u ) = f ,$ where the random field $a ( \pmb { x } , \cdot )$ models uncertain permeability variations and ω is therefore unknown.

Within this stochastic setting, the objective of surrogate modeling changes fundamentally. Rather than approximating a deterministic parameter-to-solution map, one seeks to characterize the conditional probability distribution of the solution field, namely $\mu \mapsto { \mathsf { p } } ( u | \mu )$ , where ${ \mathsf p } ( u | \mu )$ denotes the conditional law of u given µ. Equivalently, ${ \mathsf p } ( u | \mu )$ may be interpreted as the push-forward measure of P through the map ω $\mathbf { \Pi } \cdot \mapsto u ( \cdot , \pmb { \mu } , \omega )$ . After discretization, this corresponds to learning conditional probability distributions in very high-dimensional spaces, making the direct construction of probabilistic surrogates particularly challenging.

In this work, we propose tackling problems such as (2) by combing generative artificial intelligence (AI) and reduced order modeling. The former is meant to account for the stochasticity, while the latter can efectively handle the high dimensions into play. Our research is therefore part of a broader emerging trend concerning the development of probabilistic surrogates. We mention, for instance, the work by Coscia et al. [13], where generative adversarial neural networks are incorporated within ROM pipelines in order to quantify output uncertainties. There, however, no intrinstic stochasticity is present as part of the phenomenon under study: rather, all uncertainties emerge from the very definition of the ROM, essentially capturing a ROM-FOM closure in a probabilistic sense. More aligned with our setting are, instead, the recent works in [7, 41, 36, 2, 30]. These, in fact, explore the use of generative AI and machine learning for the construction of probabilistic neural operators and surrogate models, where stochasticity comes from partial knowledge of the system. Our work aims at pursuing a similar objective, however avoiding the use of overly complex architectures (such as difusion models), in addition restoring a strong connection with the ROM literature. Indeed, it is now widely recognized that reduced order modeling and operator learning are closely interconnected, providing complementary perspectives with diferent advantages and trade-ofs: we mention, for instance, the eloquent analogy between the POD-NN method [29] and the so-called POD-DeepONet architecture [38], just to mention two widely used techniques appeared in the community in the last decade.

Before presenting our approach, it is important to clarify that, despite the similarity in terminology, the setting considered in this work is fundamentally diferent from that considered in several existing contributions. An example is provided by the stochastic reduced basis (RB) methods [43], and, more generally, by most applications of the RB method to parametric PDEs with stochastic coeficients: see, e.g., [5]. There, in fact, stochasticity typically enters the governing equations in a fully specified manner, making the construction of the reduced-order model relatively straightforward. Similar approaches can be found, e.g., in [8, 9, 10]. By contrast, our central assumption is that the occurring event ω, and more generally model (2), are efectively unknown. This is meant to reflect the reality of many experimental settings, where the underlying physical mechanisms are only partially understood and the governing laws are not directly accessible. Our goal, in fact, is to lay the foundations for a unified framework that is suficiently flexible to support learning from both computational models (FOMs) and empirical data. The objectives of the aforementioned works, instead, were intrinsically diferent.

Similarly, our definition of probabilistic reduced order model is not to be confused with the one adopted by some other authors: see, e.g., [3], where the word «probabilistic» does not refer to the ROM per se, but rather relates to the algorithm underlying its construction, which the authors exploit to improve stability and generalization. In particular, these are works in which stochasticity is introduced only during the ofline stage, but disappears at inference time.

## 1.1 Our contribution

In this work, we propose POD-CNF, a probabilistic reduced-order modeling framework for high-dimensional systems arising from parametrized stochastic PDEs. The proposed approach combines Proper Orthogonal Decomposition (POD) and Conditional Normalizing Flows (CNF). The former is employed to extract a low-dimensional representation of the solution manifold from a suitable set of high-fidelity snapshots (training data). The latter, instead, is trained over the same data to approximate the conditional probability distribution of the reduced coeficients given the input parameters µ. In this manner, the surrogate model learns a parameter-to-distribution map in terms of a set of reduced coordinates, enabling the generation of complex stochastic solution realizations while maintaining computational eficiency. From a theoretical standpoint, our construction is rigorously grounded on a suitable error decomposition formula, which explicitly decouples the inaccuracies introduced by the POD basis and those caused by the CNF module. This is obtained by combining standard results on the POD projection error with a novel proof about the universality of CNFs.

The remainder of the paper is organized as follows. In Section 2, we introduce the mathematical setting and dis cuss the probabilistic formulation of reduced-order surrogate modeling for stochastic parametrized systems. Section 3 presents the proposed POD-CNF methodology, including the projection-based reduction step and the conditional generative modeling framework, discussing its underlying theoretical rationale and its practical implementation. Numerical experiments assessing the accuracy and eficiency of the proposed approach are presented in Section 4. In Section 5, we demonstrate a proof-of-concept application to forward and inverse uncertainty quantification. Finally, conclusions and future research directions are discussed in Section 6.

## 2 Problem formulation

Let $\Omega \subset \mathbb { R } ^ { d }$ be a bounded spatial domain. We consider a time-independent physical field u whose behavior is governed by a parametrized stochastic PDE (SPDE), namely

$$
\tilde { \mathcal { L } } ( \boldsymbol { u } , \boldsymbol { \mu } , \omega ) = \tilde { f } ( \boldsymbol { \mu } , \omega ) \qquad \mathrm { i n } ~ \Omega ,\tag{3}
$$

where $\pmb { \mu } \in \mathcal { P } \subset \mathbb { R } ^ { p }$ is a suitable parameter vector, taking values in a given compact set ${ \mathcal { P } } ,$ whereas ω represents a stochastic event in some probability space $( \mathcal { E } , \mathcal { F } , \mathbb { P } )$ . Here, $\tilde { \mathcal { L } } : V \times \mathcal { P } \times \mathcal { E } \stackrel { \cdot } {  } \mathbb { R }$ is a map acting as a diferential operator in the first variable, where $V \subset L ^ { 2 } ( \Omega )$ is a suitable function space —e.g., $V = H ^ { 1 } ( \Omega )$ , or $V = H _ { 0 } ^ { 1 } ( \Omega )$ and the solution to (3) exists unique for each parametric instance $\pmb { \mu }$ and each event ω. Similarly, <sup>˜</sup>f represents a parametrized stochastic source, incorporating external forces and boundary data.

We assume that the solutions to the SPDE can be either approximated numerically or measured experimentally over a suitable discretization of the spatial domain Ω, independent of $\pmb { \mu }$ and $\omega .$ . Under both scenarios, every parametric instance $\pmb { \mu } \in \mathcal { P }$ can now be associated to a corresponding high-dimensional random vector, ${ \mathbf { u } } _ { \pmb { \mu } } \in \mathbb { R } ^ { N _ { h } }$ , where $N _ { h }$ are the degrees of freedom associated with either the resolution of the data or the discretization of the numerical solver. In general, assuming a Lagrangian discretization in the second case, this translates into

$$
\begin{array} { r } { \mathrm { u } _ { \pmb { \mu } } ^ { ( j ) } = u ( \pmb { x } _ { j } , \pmb { \mu } , \cdot ) + \epsilon ( \pmb { x } _ { j } , \pmb { \mu } , \cdot ) , \qquad j = 1 , \ldots , N _ { h } , } \end{array}
$$

where $x _ { j }$ is the jth dof in the discretization and the dependence on ω has been made implicit. Here, $\epsilon = \epsilon ( x , \pmb { \mu } , \omega )$ is a further source of uncertainty, potentially associated with, e.g., measurement noise or numerical errors.

We collect all sources of uncertainty within a suitable random vector of hidden latent variables, θ. Informally, $\theta \cong [ \omega , \epsilon ]$ . More precisely, $\pmb { \theta }$ encapsulates all the variability in the discrete solution u that cannot be explained in

terms of the sole problem parameters $\pmb { \mu } .$ . We will thus write $\mathbf { u } _ { \mu } = \mathbf { u } _ { \mu } ( \pmb \theta )$ whenever willing to emphasize such stochastic dependency. In the case of numerical simulations, the vector $\mathbf { u } _ { \mu }$ can be characterized as the solution to a suitable (nonlinear) system of algebraic equations,

$$
\mathbf { L } ( \mathbf { u } , \pmb { \mu } , \pmb { \theta } ) = \mathbf { f } ( \pmb { \mu } , \pmb { \theta } )\tag{4}
$$

obtained by discretizing (3) using, $\mathrm { e . g . }$ , the FEM. With this notation, it is evident that u can be seen as the solution to a parametric problem in the pair $( \mu , \theta )$ . The issue is that θ is efectively unknown, or unobserved, which causes the solution to be stochastic. In general, the dimension of θ is also unknown. Still, it can be upper bounded by $N _ { h }$ using standard arguments from Optimal Transport theory (cf. Section 2.1).

We emphasize that the presented framework is general enough to encompass many diferent scenarios. In particular, it can be used to address problems in which the stochasticity arises from either (i) hidden physical parameters, (ii) numerical errors, (iii) measurement uncertainty, or a combination of the three. In this work, we will focus exclusively on the first scenario, leaving interactions with the other research strands for future explorations. This choice is motivated by two main reasons. The first one, is that an exhaustive treatment of problems with hidden variables is a fundamental prerequisite for scaling the framework to challenging real-world settings. In fact, experimental settings are often characterized by partial knowledge of the system and its governing equations. Furthermore, the set of explanatory variables is often underestimated, meaning that, among all of the relevant variables influencing the system, only a few are efectively observed and recorded (here corresponding to the vector of parameters µ).

The second reason is that, in contrast to (i), the literature on (ii) and (iii) is already quite mature. For instance, the research on Probabilistic Numerics [26, 11] and Statistical Solutions [18, 17] provide a solid foundation for addressing (ii). The former, in fact, allows to complement classical a priori and a posteriori estimates with rigorous uncertainty quantification on numerical errors. The latter, instead, provides a tailored framework for studying turbulent and chaotic systems, where even mild changes in the numerical solver setup can produce remarkably diferent results.

Similarly, the broad literatures on Bayesian Statistics [22, 11] and Data Assimilation [34] provide a well-established background for studying problems subject to measurement noise, as in (iii).

In light of these considerations, the present work focuses on problems where stochasticity arises from hidden, unobserved, problem parameters. Precisely, our goal is to develop a computational pipeline for constructing a probabilistic surrogate capable of capturing the complexity of the parameter-to-random-solution map, $\mu \mapsto \mathbf { u } _ { \mu } .$ , associated with the problem under study. This idea is made rigorous in the next subsection.

## 2.1 Probabilistic surrogates of parameter-to-random-solution maps

There are at least two equivalent ways of formulating our learning problem: one in terms of conditional probabilit distributions, and the other in terms of maps whose outputs are random variables. In this work, we adopt the latter perspective, which we find more convenient at inference time: see Remark 1 for a deeper discussion on this matter.

In order to set our framework, we shall first introduce some notation. Let $\mathcal { W } _ { 1 }$ denote the Wasserstein-1 distance. We recall that, given two probability measures P and $\mathbb { Q }$ over $\mathbb { R } ^ { m }$ , the latter is defined as

$$
\mathcal W _ { 1 } ( \mathbb { P } , \mathbb { Q } ) : = \operatorname* { i n f } _ { \gamma \in \Gamma ( \mathbb { P } , \mathbb { Q } ) } \int _ { \mathbb { R } ^ { m } \times \mathbb { R } ^ { m } } | x - y | \gamma ( d x d y ) ,\tag{5}
$$

where $\Gamma ( \mathbb { P } , \mathbb { Q } )$ is the set of all probability measures over $\mathbb { R } ^ { m } \times \mathbb { R } ^ { m }$ having P and $\mathbb { Q }$ as marginals, respectively. With little abuse of notation, given two random variables $X , Y \in \mathbb { R } ^ { m }$ with probability laws $\mathbb { P } _ { X }$ and $\mathbb { P } _ { Y }$ , we write $\mathcal { W } _ { 1 } ( X , Y )$ to intend $\mathcal { W } _ { 1 } ( \mathbb { P } _ { X } , \mathbb { P } _ { Y } )$ . Note that the random variables X and Y need not be defined over the same probability space for the discrepancy $\mathcal { W } _ { 1 } ( X , Y )$ to be well defined. Given any positive $q \in \mathbb { N }$ , let $\mathcal { N } ( 0 , \mathbf { I } _ { q } )$ denote the standard q-dimensional Gaussian distribution. We recall that, by standard results in Optimal Transport theory, for every random variable $X \in \mathbb { R } ^ { m }$ there exists an index $1 \leq q \leq m$ and a measurable map $T : \mathbb { R } ^ { q }  \mathbb { R } ^ { m }$ such that, if $\mathscr { z } \sim \mathcal { N } ( 0 , \mathbf { I } _ { q } )$ , then $X$ and $T ( z )$ have the same law, meaning $\mathcal { W } _ { 1 } ( X , T ( z ) ) = 0 :$ see, e.g., [52, Chapter 1]. Typically, $q = m$ . However, smaller values may at times be possible, for instance when the support of X lies inside some submanifold of $\mathbb { R } ^ { m }$

With this notation, we define a probabilistic surrogate of $\mu \mapsto \mathbf { u } _ { \mu }$ to be any map of the form

$$
\begin{array} { r } { \Phi : \mathbb { R } ^ { p } \times \mathbb { R } ^ { q }  \mathbb { R } ^ { N _ { h } } \quad \mathrm { w i t h } \ 1 \leq q \leq N _ { h } , } \end{array}\tag{6}
$$

for which $\mathcal { W } _ { 1 } ( { \mathbf { u } } _ { \mu } , \ \Phi ( \pmb { \mu } , z ) )$ is suficiently small, where $\boldsymbol { z } \sim \mathcal { N } ( 0 , \mathbf { I } _ { q } )$ and $\pmb { \mu }$ varies in $\mathcal { P } _ { \cdot }$ . Here, z is auxiliary random vector of latent variables, which is introduced to make the output of $\Phi = \Phi ( \mu , z )$ aleatoric. Intuitively, this construction aims at mimicking that of the ground truth model, $\mathbf { u } = \mathbf { u } _ { \boldsymbol { \mu } } ( \boldsymbol { \theta } )$ . However, the vectors z and θ need not to be related one to another. What matters, is that they generate (approximately) the same stochasticity in the final output. From a quantitative point of view, if the parameter space is equipped with a suitable density function $\xi = \xi ( \mu )$ , the accuracy of a probabilistic surrogate can be measured via the Average Wasserstein Distance (AWD), namely

$$
\operatorname { A W D } ( \Phi ) : = \int _ { \mathcal { P } } \mathcal { W } _ { 1 } ( \mathbf { u } _ { \mu } , \Phi ( \pmb { \mu } , z ) ) \xi ( \pmb { \mu } ) d \pmb { \mu } .\tag{7}
$$

At inference time, such probabilistic surrogates can be used to generate multiple plausible solutions in real-time. For instance, given an operating condition of interest $\mu _ { 0 } \in \mathcal { P }$ , one can produce multiple independent random solutions $\tilde { \mathbf { u } } _ { \mu _ { 0 } , 1 } , \dots , \tilde { \mathbf { u } } _ { \mu _ { 0 } , m _ { \mathrm { g e n } } }$ by sampling $z _ { 1 } , \ldots , z _ { m _ { \mathrm { g e n } } } \sim { \mathcal { N } } ( 0 , \mathbf { I } _ { q } )$ and computing $\tilde { \mathbf { u } } _ { \mu _ { 0 } , k } = \boldsymbol { \Phi } ( \mu _ { 0 } , z _ { k } )$ . These realizations can then be used to approximate the conditional law of $\mathbf { u } _ { \mu }$ and to estimate relevant quantities of interest. If the samples are very similar to each other then one can conclude that, under $\pmb { \mu } _ { 0 }$ , the efect of the hidden parameters is nearly negligible. In contrast, large deviations or multi-modal behaviors can be taken as evidence of intrinsic uncertainties afecting the system.

Remark 1 (Learning generators vs learning distributions). Since we are only interested in the conditional law of the solution field $\mathbf { u } _ { \mu }$ , and not in the random variable per se, an alternative approach is to define a probabilistic surrogate as a (nonlinear) operator mapping parameters onto probability distributions, $\mathcal { P } \mapsto \mathcal { P } ( \mathbb { R } ^ { N _ { h } } )$ . However, since $N _ { h }$ can be fairly large, constructing an object that returns explicit probability distributions over R<sup>Nh</sup> can be highly nontrivial. Furthermore, at inference time, using such a tool can pose significant challenges. Consider for instance the problem of employing the surrogate to estimate a given statistics of the solution, $e . g . , \mathbb { E } [ Q ( \mathbf { u } _ { \mu _ { 0 } } ) ]$ for some $Q : \mathbb { R } ^ { N _ { h } }  \mathbb { R }$ and $\mu _ { 0 } \in \mathcal { P }$ . In this case, the approximation would read

$$
\mathbb { E } [ Q ( \mathbf { u } _ { \pmb { \mu } _ { 0 } } ) ] \approx \int _ { \mathbb { R } ^ { N _ { h } } } Q ( \mathbf { u } ) \hat { \mathbb { P } } _ { \pmb { \mu } _ { 0 } } ( d \mathbf { u } ) ,\tag{8}
$$

where $\hat { \mathbb { P } } _ { \pmb { \mu } _ { 0 } } \in \mathcal { P } ( \mathbb { R } ^ { N _ { h } } )$ is the probability law predicted by the surrogate. However, even $i f \operatorname { \hat { \mathbb { P } } } _ { \mu _ { 0 } }$ were available in closed form, for example through its density, evaluating the above integral would remain computationally challenging: numerical quadrature methods do not scale suficiently well to be practical in $\mathbb { R } ^ { N _ { h } }$ , while Monte Carlo methods can easily become troublesome in high-dimensions, even when the underlying density is known explicitly: see, $e . g .$ , Markov-Chain Monte Carlo samplers and similar tools [22, Chapters 10-12].

In contrast, learning a probabilistic surrogate that operates as a generative model greatly simplifies the problem. In fact, the generative model directly acts as an approximate conditional sampler, allowing one to estimate expectations via Monte Carlo sampling, e.g.,

$$
\mathbb { E } [ Q ( \mathbf { u } _ { \mu _ { 0 } } ) ] \approx \frac { 1 } { m _ { g e n } } \sum _ { k = 1 } ^ { m _ { g e n } } Q ( \Phi ( \pmb { \mu _ { 0 } } , z _ { k } ) ) .\tag{9}
$$

In general, the accuracy of the above will depend on the quality of the surrogate and on the number of Monte Carlo samples. However, we notice that this second source of inaccuracy can be made arbitrarily small at a negligible computational cost: in fact, the Monte Carlo error scales as $O ( m _ { g e n } ^ { - 1 / 2 } )$ , but because the surrogate is computationally cheap to evaluate, $m _ { g e n }$ can indeed be taken as large as required.

## 3 The POD-CNF approach

We assume to have access to a dataset $\{ \mu _ { i } , \mathbf { u } _ { i } \} _ { i = 1 } ^ { N }$ consisting of diferent parametric configurations and corresponding random solutions. These may either consist of numerical simulations, obtained by means of a given stochastic FOM, or derive from suitable real-world experiments. We stress that, in both cases, the system under study is assumed to be time-independent, but potentially subject to complex physical laws that entail nonlinearities. We also remark that $\mathbf { u } _ { i } = \mathbf { u } _ { \mu _ { i } } ( \pmb \theta _ { i } )$ , meaning that the i-th solution in the dataset corresponds to the i-th parametric configuration $( \pmb { \mu } _ { i }$ , which is known) and the i-th stochastic event $( \pmb \theta _ { i }$ , which is unknown). Our goal is to leverage this data in order to construct a surrogate Φ of the parameter-to-random-solution map $\mu \mapsto \mathbf { u } _ { \mu }$ . To this end, we propose the following ansatz,

$$
\Phi ( \pmb { \mu } , z ) : = \mathbf { V } \phi ( \pmb { \mu } , z )\tag{10}
$$

which is based on a low-rank representation of the solution field. More precisely, for a suitable reduced dimension $n \ll N _ { h }$ , we assume that:

$\mathbf { V } \in \mathbb { R } ^ { N _ { h } \times n }$ is a deterministic orthonormal matrix, whose purpose is to reduce the dimensionality of the problem. The idea, in fact, is that although the solutions u live in a high dimensional space, $\mathbb { R } ^ { N _ { h } }$ , there may exist a much smaller space, $\gamma = \operatorname { s p a n } ( \mathbf { V } ) \cong \mathbb { R } ^ { n }$ , in which the solutions approximately lie. Informally, if we denote by $\operatorname { s u p p } ( \mathbf { u } _ { \mu } )$

the support of the random solution $\mathbf { u } _ { \mu } .$ , this corresponds to saying that $\cup _ { \pmb { \mu } \in \mathcal { P } ^ { \mathrm { S u p p } } } ( \mathbf { u } _ { \pmb { \mu } } )$ is approximately contained in V. From a computational standpoint, if the parameter space is endowed with a suitable density $\xi = \xi ( \mu )$ , such degree of approximation can be quantified as

$$
\int _ { \mathcal { P } } \mathbb { E } \| \mathbf { u } _ { \mu } - \mathbf { V } \mathbf { V } ^ { \top } \mathbf { u } _ { \mu } \| \xi ( \mu ) d \mu ,\tag{11}
$$

i.e. by the average expected projection error;

$\phi : \mathbb { R } ^ { p } \times \mathbb { R } ^ { n } \to \mathbb { R } ^ { n }$ is a latent generative model – corresponding to $q = n$ in Eq. (6). The latter aims at capturing both the variability induced by the problem parameters, $\mu ,$ as well as the one produced by the hidden stochastic variables, θ. Intuitively, one aims at constructing $\phi$ in such a way that, for any given $\pmb { \mu } \in \mathcal { P }$ , the distribution of the generative model, $\phi ( \pmb { \mu } , z )$ with $z \sim \mathcal { N } ( 0 , \mathbf { I } _ { n } )$ , is close to that of the random projection coeficients, $\mathbf { V } ^ { \top } \mathbf { u } _ { \mu }$ In analogy to Eq. (7), the quality of such approximation can be measured using, e.g., the AWD metric.

From a practical standpoint, both the construction of the matrix V and that of the latent generative model $\phi$ will be based on the data at hand. Specifically, in this work we propose relying on the Principal Orthogonal Decomposition (POD) algorithm and on Conditional Normalizing Flow (CNF) architectures in order to construct V and ϕ, respectively, reason for which we term our approach POD-CNF.

We mention that, in spirit, our proposal can also be understood as a stochastic generalization of the already wellestablished POD-NN method [29]. The latter, in fact, is a data-driven ROM strategy, developed for deterministic problems, in which the solution is approximated as $\mathbf { u } _ { \mu } \approx \mathbf { V } \tilde { \phi } ( \mu )$ , with V the POD matrix and $\tilde { \phi } : \mathbb { R } ^ { p }  \mathbb { R } ^ { n }$ a standard multilayer perceptron (MLP) network trained to approximate the parameter-to-coeficient map $\mu \mapsto \mathbf { V } ^ { \top } \mathbf { u } _ { \mu }$ . Since its introduction, the POD-NN method has had a significant impact on the literature, with several subsequent approaches building upon or tracing their origins back to this pioneering idea. For instance, the DeepONet architecture [38] can be seen as a continuous generalization of the POD-NN; similarly, alternative designs and implementations have led to the development of other prominent strategies such as POD-DL-ROM [21], Conservative POD-NN [4], DOD-NN [20, 33] and many more [53, 46]. In this sense, the POD-NN approach can be regarded as the forerunner of a much wider class of methods, which our contribution aims to further extend.

The remainder of this Section is devoted to establishing suitable theoretical results supporting our construction (§3.1), as well as to presenting its practical implementation from an algorithmic standpoint (§3.2).

## 3.1 Theoretical rationale

There are several reasons for using POD and CNFs. To start, POD is a powerful and well-established algorithm, which is known to enjoy remarkable optimality properties. Furthermore, its implementation is relatively straightforward, as it boils down to a truncated Singular Value Decomposition (SVD) of the snapshots matrix $\mathbf { S } = [ \mathbf { u } _ { 1 } , \dots , \mathbf { \bar { u } } _ { N } ] \in \mathbb { R } ^ { N _ { h } \times N }$ On top of this, POD allows for the construction of stable ROMs, in that small errors in the reduced space are propagate isometrically to the full order space via the map $\mathbb { R } ^ { n } \ni \mathbf { c } \mapsto \mathbf { V } \mathbf { c } \in \mathbb { R } ^ { N _ { h } }$ . Finally, by construction, POD tends to filter out high frequency modes, which can be helpful in removing external white noise-type fluctuations in the data.

The use of CNFs, instead, is mainly motivated by their expressivity and their computational eficiency. Indeed, under fairly mild assumptions, normalizing flows are known to be universal approximators [32], which makes them good candidates for learning the latent generative model $\mu \mapsto \mathbf { V } ^ { \intercal } \mathbf { u } _ { \mu }$ . At the same time, they are also: (i) computationally much less expensive to train than modern score-based generative models, such as Difusion Models [50], (ii) far more eficient than other flow-based architectures, such as Flow Matching [37, 1] or Stochastic Interpolants [1, 42], when employed at inference time, which makes them better suited for many query applications.

By combining the universality of CNFs with classical results on POD truncation errors, it is then relatively straightforward to derive corresponding theoretical insights for POD-CNF. This connection is formalized in the next two paragraphs, where we state several foundational results along this direction. Right after that, a brief discussion on the limitations of our approach is also provided. In order to keep the paper self-contained, technical proofs are deferred to the Appendix A.

Proper Orthogonal Decomposition (POD): idea and motivation. Proper Orthogonal Decomposition (POD) is a powerful data-driven technique which commonly used for dimensionality reduction and reduced order modeling in general. Its primary objective is to extract the most dominant characteristics (often called "modes") from a highdimensional dataset. By identifying these modes, POD allows for a low-dimensional approximation that captures the essential behavior of the original system while drastically reducing its complexity. It is best suited in those frameworks in which linear reduction is efective, cf. [48, 12].

In the simplest scenario, the implementation of the POD consists of computing an SVD of the snapshots matrix ${ \bf S } = [ { \bf u } _ { 1 } , \dots , { \bf u } _ { N } ] \in \mathbb { R } ^ { N _ { h } \times N }$ , namely

$$
\mathbf { S } = \mathbf { U } \Sigma \mathbf { W } ^ { \top } ,
$$

where $\mathbf { U } \in \mathbb { R } ^ { N _ { h } \times N _ { h } }$ and $\mathbf { W } \in \mathbb { R } ^ { N \times N }$ are orthogonal matrices containing the left and right singular vectors, respectively, while Σ is a diagonal matrix containing the singular values, $\sigma _ { 1 } \geq \sigma _ { 2 } \geq \cdot \cdot \cdot \geq 0$ , sorted in decreasing order. The POD matrix $\mathbf { V } \in \mathbb { R } ^ { N _ { h } \times n }$ is then computed by retaining the first n columns of U. It is well-known that this procedure is empirically optimal, in that

$$
\sum _ { i = 1 } ^ { N } \| \mathbf { u } _ { i } - \mathbf { V } \mathbf { V } ^ { \top } \mathbf { u } _ { i } \| ^ { 2 } = \operatorname* { m i n } _ { \mathbf { Q } \in \mathbb { R } ^ { N _ { h } \times n } } \sum _ { i = 1 } ^ { N } \| \mathbf { u } _ { i } - \mathbf { Q Q } ^ { \top } \mathbf { u } _ { i } \| ^ { 2 } = \sum _ { i > n } \sigma _ { i } ^ { 2 } ,\tag{12}
$$

see, e.g., [48]. In other words, the POD basis is optimal in the mean-square sense, minimizing the reconstruction error of the computed snapshots among all subspaces of a prescribed dimension. If the parameter instances $\mu _ { i } \in \mathcal P$ are drawn independently according to $f ,$ and the events $\pmb \theta _ { i }$ are independent as well, then, in the infinite data limit, V is known to converge to the ideal POD matrix $\mathbf { V } _ { \infty } .$ solution to the minimization problem below

$$
\operatorname* { m i n } _ { \mathbf { Q } \in \mathbb { R } ^ { N _ { h } \times n } } \int _ { \mathcal { P } } \mathbb { E } \| \mathbf { u } _ { \mu } - \mathbf { Q Q } ^ { \top } \mathbf { u } _ { \mu } \| ^ { 2 } \boldsymbol { \xi } ( \mu ) d \mu ,\tag{13}
$$

see, $\mathrm { e . g . }$ , [6, Section 2.2]. Theoretically, much like V is built by computing an SVD of S, the ideal POD matrix can be computed by considering an SVD of the Hilbert-Schmidt operator $\begin{array} { r } { \mathcal { S } : \bar { L } ^ { 2 } ( \mathcal { P } \times \mathbb { R } ^ { n } ) \ni g \mapsto \int _ { \mathcal { P } } \mathbb { E } [ \mathbf { u } _ { \mu } g ] \xi ( \pmb { \mu } ) d \pmb { \mu } \in \mathbb { R } ^ { N _ { h } } } \end{array}$ Equivalently, $\mathbf { V } _ { \infty }$ can also be derived by diagonalizing the uncentered covariance matrix,

$$
\mathbf { C } : = \int _ { \mathcal { P } } \mathbb { E } [ \mathbf { u } _ { \pmb { \mu } } \otimes \mathbf { u } _ { \pmb { \mu } } ] \xi ( \pmb { \mu } ) d \pmb { \mu } \in \mathbb { R } ^ { N _ { h } \times N _ { h } } ,\tag{14}
$$

and then collecting the n eigenvectors associated with the n largest eigenvalues $\lambda _ { i }$

In general, if $\sigma _ { i , \infty }$ are the singular values of $\mathcal { S }$ in the infinite data-limit, and $\sigma _ { i , N }$ are those of S obtained with N randomly drawn snapshots, then one has

$$
\operatorname* { l i m } _ { N  + \infty } \frac { 1 } { N } \sigma _ { i , N } ^ { 2 } = \sigma _ { i , \infty } ^ { 2 } = \lambda _ { i } ,
$$

almost surely: see, for instance, [6, Section 2.2] and [28, Chapter 3]. The whole machinery can also be generalized very easily to account for other inner-product norms, diferent from the plain Euclidean one, cf. [48, 28].

In our setting, the use of a projection-based dimensionality reduction technique is motivated by the following decoupling lemma, which shows how the error a generative ROM built according to our ansatz can be split into two contributions: (i) a projection error, and (ii) a latent generative error.

Lemma 1. Let $\{ \mathbf { u } _ { \mu } \} _ { \mu \in \mathcal { P } }$ be a parametric family of random variables in $\mathbb { R } ^ { N _ { h } }$ , with $\mathcal { P } \subseteq \mathbb { R } ^ { p }$ , and let $\mathbf { V } \in \mathbb { R } ^ { N _ { h } \times n }$ be an orthonormal matrix. Let $\phi : \mathbb { R } ^ { p } \times \mathbb { R } ^ { n } \to \mathbb { R } ^ { n }$ be given, and let $z \in \mathbb { R } ^ { n }$ be a random vector with law $z \sim \mathbb { Q }$ . Then,

$$
\mathcal { W } _ { 1 } ( \mathbf { u } _ { \mu } , \mathbf { \Delta } \mathbf { V } \phi ( \mu , z ) ) \leq \mathbb { E } \| \mathbf { u } _ { \mu } - \mathbf { \Delta } \mathbf { V } \mathbf { V } ^ { \top } \mathbf { u } _ { \mu } \| + \mathcal { W } _ { 1 } \big ( \mathbf { V } ^ { \top } \mathbf { u } _ { \mu } , ~ \phi ( \mu , z ) \big ) \qquad \forall \mu \in \mathcal { P } .\tag{15}
$$

Proof. See Appendix A.

Combining the result above with the aforementioned optimality of POD immediately yields the following corollary.

Corollary 1. Let $\{ \mathbf { u } _ { \mu } \} _ { \mu \in \mathcal { P } }$ be a parametric family of random variables in $\mathbb { R } ^ { N _ { h } }$ , with $\mathcal { P } \subseteq \mathbb { R } ^ { p }$ . Let f be a probability density over P. Denote by C the $N _ { h } \times N _ { h }$ positive semi-definite matrix in (14). Let $\lambda _ { 1 } \geq \dots \geq \lambda _ { N _ { h } } \geq 0$ be the eigenvalues of C, listed in decreasing order, and let $\mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { N _ { t } }$ be the corresponding orthonormal eigenvectors. Given any truncation index $1 \leq n \leq N _ { h }$ , assemble the orthonormal matrix $\mathbf { V } = [ \mathbf { v } _ { 1 } , \dots , \mathbf { v } _ { n } ] \in \mathbb { R } ^ { N _ { h } \times n }$ . For any $\phi : \mathbb { R } ^ { p } \times \mathbb { R } ^ { n } $ $\mathbb { R } ^ { n }$ and any n-dimensional random vector z, one has

$$
\int _ { \mathcal { P } } \mathcal { W } _ { 1 } ( \mathbf { u } _ { \mu } , \mathbf { V } \phi ( \mu , z ) ) \xi ( \mu ) d \mu \leq \left[ \sum _ { j = n + 1 } ^ { N _ { h } } \lambda _ { j } \right] ^ { 1 / 2 } + \int _ { \mathcal { P } } \mathcal { W } _ { 1 } ( \mathbf { V } ^ { \top } \mathbf { u } _ { \mu } , \phi ( \mu , z ) ) \xi ( \mu ) d \mu .\tag{16}
$$

Proof. See Appendix A.

Basically, Corollary 1 ensures that if the eigenvalues $\lambda _ { i }$ decay suficiently fast, then one can construct an efective probabilistic ROM by doing a POD and transferring all of the stochasticity at the latent level, in analogy to what is commonly done in the deterministic case.

Conditional Normalizing Flows (CNF): idea and motivation. Conditional Normalizing Flows (CNFs) are a class of generative models that aim to approximate conditional densities by leveraging optimal transport theory and the change of variables formula. Simply put, in order to approximate the conditional density of a given random variable, $\mathbf { y } | \mu ,$ CNFs progressively transform Gaussian noise into the desired distribution by applying a sequence of µ-dependent transformations, $T _ { \mu , 1 } , \ldots , T _ { \mu , K }$ . Such transformations are designed to be invertible in their main argument. This allows making the density of the transformed vector, $( T _ { \pmb { \mu } , K } \circ \dotsb \circ T _ { \pmb { \mu } , 1 } ) ( z )$ , fully explicit; in fact, thanks to the change of variables formula, one has

$$
f _ { ( T _ { \mu , K } \circ \cdots \circ T _ { \mu , 1 } ) ( z ) } ( \mathbf { y } ) = f _ { z } { \big ( } ( T _ { \mu , 1 } ^ { - 1 } \circ \cdots \circ T _ { \mu , K } ^ { - 1 } ) ( \mathbf { y } ) ) \cdot \prod _ { k = 1 } ^ { K } \left| \operatorname* { d e t } J _ { T _ { \mu , k } } ^ { - 1 } ( ( T _ { \mu , k } ^ { - 1 } \circ \cdots \circ T _ { \mu , K } ^ { - 1 } ) ( \mathbf { y } ) ) \right| ,\tag{17}
$$

with $f _ { z }$ the density of the standard normal distribution in $\mathbb { R } ^ { n }$

From a practical perspective, the transformations $T _ { \mu , 1 } , \ldots , T _ { \mu , K }$ constitute the trainable components of the CNF architecture, and they are usually designed by cleverly combining diferent units based on multi-layer perceptrons (MLPs). Such design is crucial, as it prescribes the expressivity of the whole architecture, but it also entails some nontrivial challenges. In fact, in order to make (17) practical, one must design the maps $T _ { \mu , k }$ so that they are easily invertible and so that Jacobian determinants can be computed eficiently.

In this respect, an approach which has proven successful, and which we adopt in this work, is to construct the transformations $T _ { \mu , k }$ by relying on afine coupling layers. The latter were first introduced by Dinh et al. in [15] as part of a normalizing flow architecture thereon referred to as RealNVP. Here, we shall consider a conditional variant of such architecture, which can be described as follows. For every $k \in \{ 1 , \ldots , K \}$ , let $z _ { k - 1 }$ denote a generic input for the k-th layer. The vector is split into two halves, $z _ { k - 1 , A _ { k } }$ and $z _ { k - 1 , B _ { k } }$ , with $A _ { k } \cup B _ { k } = \{ 1 , . . . , n \}$ being a suitable partition of the vector indices, $A _ { k } \cap B _ { k } = \emptyset$ . Such partition is typically chosen at random and varies from layer to layer. If n is even, a standard choice is to set $A _ { k } = \{ 1 , \dots , n / 2 \}$ and $B _ { k } = \{ n / 2 + 1 , \ldots , n \}$ when k is even, and $A _ { k } = \{ n / 2 + 1 , \ldots , n \}$ and $B _ { k } = \{ 1 , \ldots , n / 2 \}$ when k is odd. In this case, the RealNVP is said to consist of alternating afine coupling layers.

With this convention, the action of the kth layer $T _ { \mu , k }$ reads

$$
T _ { \mu , k } ( z _ { k - 1 } ) = \left\{ \begin{array} { l l } { z _ { k , A _ { k } } = z _ { k - 1 , A _ { k } } } \\ { z _ { k , B _ { k } } = z _ { k - 1 , B _ { k } } \odot \exp ( { s _ { k } ( z _ { k - 1 , A _ { k } } , \mu ) } ) + t _ { k } ( z _ { k - 1 , A _ { k } } , \mu ) } \end{array} \right.\tag{18}
$$

where $s _ { k } , t _ { k } : \mathbb { R } ^ { | A _ { k } | + p }  \mathbb { R } ^ { | B _ { k } | }$ are trainable MLPs, referred to as the scale and translation parameter, respectively, whereas ⊙ denotes the Hadamard product. The full CNF architecture is then given by

$$
\phi ( \pmb { \mu } , z ) : = ( T _ { \pmb { \mu } , K } \circ \cdot \cdot \cdot \circ T _ { \pmb { \mu } , 1 } ) ( z ) .\tag{19}
$$

We will say that ϕ is a ReLU CNF with alternating afine coupling layers if: $( i )$ the transformations $T _ { \mu , k }$ are defined as in (18), with $A _ { k }$ and $B _ { k }$ given by the canonical splitting; $( i i )$ the MLPs $s _ { k }$ and $t _ { k }$ use the ReLU activation in all of their hidden layers; (iii) the translation $\operatorname { M L P s } , t _ { k }$ , have no terminal activation; (iv) the scale MLPs, $s _ { k } .$ , are equipped with a terminal continuous activation of choice, $\rho _ { k } : \mathbb { R } \to \mathbb { R }$ . For numerical stability, such activations are often chosen to be bounded: in our experiments, for instance, we use the hyperbolic tangent activation for all $k = 1 , \ldots , K .$ . Still, when discussing theoretical results we will always assume that $- \infty \in \overline { { \rho _ { k } ( \mathbb { R } ) } }$ , thus preventing boundness from below. This is to guarantee the universality of the resulting CNF architecture under fairly general conditions: we refer the interested reader to Remark B.1 in the Appendix.

The peculiar construction in (18), where each layer leaves one half of the input vector unchanged, while scaling and translating the other half, is what makes RealNVPs particularly efective in practice. This, in fact, guarantees that each $T _ { \mu , k } ,$ is invertible —regardless of the learned MLPs $s _ { k }$ and $t _ { k } -$ and that the Jacobian of $T _ { \mu , k } ^ { - 1 }$ is triangular, allowing for a fast computation of its determinant. At the same time, this design is general enough to make RealNVPs universal approximators: see, e.g., [32] for results in the unconditional case, or [39] and Lemma B.1 in Appendix B for suitable extensions to parametric problems.

For all such reasons, we propose relying on CNFs in order to learn the latent generative model in Eq. (10). With this choice, the aforementioned universality property, together with Corollary 1, immediately yields the following result, which provides the theoretical foundation for the POD-CNF approach.

Theorem 1. Let $\mathbf { u } _ { \mu }$ be a parametrized random vector in $\mathbb { R } ^ { N _ { h } }$ , where $\pmb { \mu } \in \mathcal { P } \subset \mathbb { R } ^ { p }$ varies over a suitable parameter space, with P compact. Let f be a probability density defined over P. Assume that the law of $\mathbf { u } _ { \mu }$ varies continuously in the $\mathcal { W } _ { 1 }$ metric with respect to $\mathbf { \Omega } _ { \mu } ,$ and that there exists some $R > 0$ such that $\| \mathbf { u } _ { \mu } \| \leq R$ almost surely for all $\pmb { \mu } \in \mathcal { P }$

Given any truncation index $n \geq q ,$ let V be as in Corollary 1. Let $\mathbb { P } _ { n , \mu }$ be the probability distribution of the reduced random vector $\mathbf { V } ^ { \top } \mathbf { u } _ { \mu } .$ . Assume that, for every $\pmb { \mu } \in \mathcal { P }$ , the distribution $\mathbb { P } _ { n , \mu }$ is absolutely continuous. Then, for every $\epsilon > 0$ , there exists a ReLU conditional normalizing flow $\phi _ { \epsilon }$ with alternating afine coupling layers such that

$$
\int _ { \mathcal { P } } { \mathcal { W } } _ { 1 } ( { \mathbf { u } } _ { \mu } , \mathbf { V } \phi _ { \epsilon } ( { \pmb { \mu } } , z ) ) \xi ( { \pmb { \mu } } ) d { \pmb { \mu } } \leq \left[ \sum _ { j = n + 1 } ^ { N _ { h } } \lambda _ { j } \right] ^ { 1 / 2 } + \epsilon .
$$

Proof. See Appendix A.

Intrinsic limitations of the POD-CNF approach. While the POD-CNF framework provides a principled and computationally eficient strategy for learning parametric solution distributions, it also relies on a number of assump tions that may limit its applicability in challenging regimes.

First, the efectiveness of the approach is intrinsically tied to the compressibility of the solution manifold by a linear subspace. In particular, the quality of the POD representation depends on the decay of the eigenvalues of the correlation operator. This is closely related to the behavior of the Kolmogorov n-width of the supporting solution set $\mathcal { S } = \cup _ { \mu \in \mathcal { P } } \mathrm { s u p p } ( \mathbf { u } _ { \mu } ) = \cup _ { \mu \in \mathcal { P } } \{ \mathbf { v } \in \mathbb { R } ^ { N _ { h } } \ : \ \forall \delta > 0 \ \mathbb { P } ( | \mathbf { u } _ { \mu } - \mathbf { v } | < \delta ) > 0 ) \}$ : when this quantity decays slowly, a large number of POD modes may be required to achieve a satisfactory approximation accuracy. In such cases, the output dimension of the CNF increases, hindering the computational advantages of the reduced representation and potentially making the training of the generative model more challenging.

A related limitation is that POD is a linear dimensionality reduction technique and, by construction, preferentially retains energetic modes while filtering out small-scale and high-frequency components. Although this property can be beneficial when the discarded fluctuations are mainly associated with noise, it may become detrimental for problems where localized phenomena, sharp interfaces, or fine-scale structures play a relevant role in the solution dynamics. In such settings, nonlinear dimensionality reduction techniques, such as autoencoder-based representations, may provide more expressive alternatives and constitute a natural direction for future investigation.

A second class of limitations concerns the choice of CNFs as generative models. Although such models provide an attractive combination of expressivity and computational eficiency, their construction is constrained by invertibility assumption. For instance, architectures based on afine coupling transformations, such as the RealNVP model adopted in this work, achieve tractable likelihood evaluation at the expense of restricting the class of admissible transformations. More flexible alternatives, including continuous normalizing flows and flow-matching approaches, alleviate some of these architectural constraints, but typically require the numerical solution of an ordinary diferential equation during inference, which can become computationally demanding in multi-query applications.

Finally, the scalability of CNFs with respect to their ambient dimension remains an important practical consideration. Even after POD truncation, the reduced random variables may exhibit highly anisotropic or nearly singular distributions, for instance when the data are concentrated around lower-dimensional structures embedded in the latent space. Since standard CNFs are constructed by transporting a full-dimensional reference distribution, accurately representing such distributions may require increasingly complex transformations and can lead to optimization dificulties. Developing dimensionality-adaptive generative models or combining POD with nonlinear manifold-learning techniques may provide efective strategies to address these challenges, representing an interesting direction for future research.

## 3.2 Practical implementation

From an operational standpoint, we construct a POD-CNF surrogate through a three-step ofline procedure, whose workflow is further summarized in Algorithm 1.

Step 1: data collection and splitting. To start, we first sample S parametric configurations $\mu _ { 1 } , \dots , \mu _ { S } \in \mathcal { P }$ by drawing independent realizations from the sampling density ξ. For each $\pmb { \mu } _ { i } .$ , we then run the FOM (resp., the lab experiment) to obtain a random solution $\mathbf { u } _ { i } = \mathbf { u } _ { \mu _ { i } } ( \pmb \theta _ { i } )$ . We then split the dataset

$$
\begin{array} { r } { \{ \pmb { \mu } _ { i } , \mathbf { u } _ { i } \} _ { i = 1 } ^ { S } = \{ \pmb { \mu } _ { i } , \mathbf { u } _ { i } \} _ { i = 1 } ^ { N } \cup \{ \pmb { \mu } _ { i } ^ { \mathrm { v a l i d } } , \mathbf { u } _ { i } ^ { \mathrm { v a l i d } } \} _ { i = 1 } ^ { N _ { \mathrm { v a l i d } } } \cup \{ \pmb { \mu } _ { i } ^ { \mathrm { t e s t } } , \mathbf { u } _ { i } ^ { \mathrm { t e s t } } \} _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } , } \end{array}
$$

into training, validation, and test sets, generally following a 75:20:5 ratio.

Step 2: POD basis extraction. We leverage the training data to assemble the snapshots matrix $\mathbf { S } = [ \mathbf { u } _ { 1 } , \dots , \mathbf { u } _ { N } ]$ and compute its SVD. In light of Eq. (12), we perform POD by truncating the SVD to the first even index $n \in 2  { \mathbb { N } }$ for which the relative residual energy,

$$
\left[ \sum _ { i > n } \sigma _ { i } ^ { 2 } \right] / \left[ \sum _ { i = 1 } ^ { \operatorname* { m i n } N _ { h } , N } \sigma _ { i } ^ { 2 } \right] ,
$$

is below a given threshold $\tau \in \mathsf { \Gamma } ( 0 , 1 )$ . We require n to be an even number in order to standardize the subsequent construction of the CNF module via alternating coupling layers, cf. §3.1.

Step 3: CNF implementation and training. To construct the CNF module, we start by choosing a depth K and designing the architectures of the trainable MLPs in the diferent coupling layers. Let us denote by $\pmb { \eta } \in \mathbb { R } ^ { D }$ the vector collecting all the trainable parameters appearing in the definition of the CNF. The trainable architecture can be then regarded as a map $\phi _ { \pmb { \eta } } : \mathbb { R } ^ { p } \times \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ , where η is to be optimized.

To train the CNF, i.e., to estimate the optimal value for η, we first assemble the reduced dataset $\{ \mu _ { i } , \mathbf { c } _ { i } \} _ { i = 1 } ^ { N }$ where $\mathbf { c } _ { i } : = \mathbf { V } ^ { \top } \mathbf { u } _ { i }$ are the POD-coeficients of the training snapshots. The model is then trained by minimizing the negative log-likelihood (NLL),

$$
{ \mathsf { N L L } } ( \pmb { \eta } ) : = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \hat { f } _ { \pmb { \eta } , \pmb { \mu } _ { i } } ( \mathbf { c } _ { i } )\tag{20}
$$

where $\hat { f } _ { \eta , \mu }$ denotes the density of $\phi _ { \eta } ( \pmb { \mu } , z )$ with $z \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { n } )$ . If we denote by $T _ { \mu , k } ^ { \eta }$ the kth coupling layer in $\phi _ { \eta } ,$ using Eq. (17), we may rewrite Eq. (20) as

$$
\small \mathrm { N L } ( \eta ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ \frac { 1 } { 2 } \| [ \phi _ { \eta } ( \mu _ { i } , \cdot ) ] ^ { - 1 } ( \mathbf { V } ^ { \top } \mathbf { u } _ { i } ) \| ^ { 2 } - \sum _ { k = 1 } ^ { K } \log \left| \operatorname* { d e t } J _ { T _ { \mu _ { i } , k } } ^ { - 1 } ( ( T _ { \mu _ { i } , K } ^ { \eta } \circ \cdot \cdot \cdot \circ T _ { \mu _ { i } , k } ^ { \eta } ) ^ { - 1 } ( \mathbf { V } ^ { \top } \mathbf { u } _ { i } ) ) \right| \right] + \mathrm { c o n s t a n t } .\tag{21}
$$

Minimizing (numerically) the above function with respect to η yields the trained architecture $\phi = \phi _ { \eta _ { \ast } }$

We mention that, in the literature, this is the standard approach for training normalizing flows and CNFs in general [44]. From a theoretical standpoint, there are several reasons that make NLL minimization a natural choice. One of such reasons is that, up to a Monte Carlo approximation, it can be shown that minimizing (20) is equivalent to minimizing the expected Kullback-Leibler divergence (EKL) between the true conditional density of $\mathbf { V } ^ { \top } \mathbf { u } _ { \mu }$ , denoted as $f _ { \mu } ,$ , and the conditional density predicted by the CNF module, that is

$$
\mathsf { E K L } ( \pmb { \eta } ) = \int _ { \mathcal { P } } D _ { \mathrm { K L } } ( f _ { \pmb { \mu } } , \hat { f } _ { \pmb { \mu } , \pmb { \eta } } ) \xi ( \pmb { \mu } ) d \pmb { \mu } .\tag{22}
$$

We refer the interested reader to, e.g., [23, Chapter 5.5] for a deeper discussion on such equivalence. In our framework, this correspondence becomes particularly valuable as it allows us to connect the CNF training with our theoretica analysis in §3.1. In fact, if we assume the two densities in Eq. (22) to be compactly supported, it is easy to see that, by Pinsker’s inequality,

$$
\begin{array} { r } { \mathcal { W } _ { 1 } ( \mathbf { V } ^ { \top } \mathbf { u } _ { \mu } , \phi ( \mu , z ) ) \leq c \| f _ { \mu } - \hat { f } _ { \mu , \eta } \| _ { L ^ { 1 } } ( \mathbb { R } ^ { n } ) \leq c \sqrt { 2 D _ { \mathrm { K L } } ( f _ { \mu } , \hat { f } _ { \mu , \eta } ) } , } \end{array}
$$

where $c > 0$ is a suitable constant depending only the sizes of the supports of the two densities. This draws a connection between the EKL and the AWD (which can be made more explicit by applying Jensen’s inequality), thus linking our theoretical analysis with the minimization of (20).

Remark 2. In practice, the design of the CNF architecture will depend on the problem under consideration. Following standard practice in machine and deep learning, we account for this implicit dependency by tuning the model and training hyperparameters—including the number of coupling layers, the complexity of the MLPs in the architecture, the learning rate, and the number of training epochs—based on NLL performance on the validation set. More precisely, given a finite collection of candidate architectures, $\phi _ { \eta } ^ { ( 1 ) } , \ldots , \phi _ { \eta } ^ { ( r ) }$ , and corresponding training routines, we train all the models separately, obtaining $\phi _ { \eta _ { * } } ^ { ( 1 ) } , \ldots , \phi _ { \eta _ { * } } ^ { ( r ) }$ , and then set $\phi = \phi _ { \eta _ { \ast } } ^ { j _ { \ast } }$ where

$$
j _ { * } : = \underset { j \in \{ 1 , \dots , r \} } { \mathrm { a r g } \mathrm { m i n } } - \frac { 1 } { N _ { v a l i d } } \sum _ { i = 1 } ^ { N _ { v a l i d } } \log \hat { f } _ { \eta _ { * } , \mu _ { i } ^ { v a l i d } } ^ { ( j ) } ( \mathbf { V } ^ { \top } \mathbf { u } _ { j } ^ { v a l i d } )
$$

with $\hat { f } _ { \eta _ { * } , \mu } ^ { ( j ) }$ being the conditional density associated to the jth model $\phi _ { \eta _ { * } } ^ { ( j ) }$

## 3.2.1 Deployment at inference time

After training, the POD-CNF becomes fully operational. Given any query parameter $\mu _ { 0 } ~ \in ~ \mathcal { P }$ , the model can efficiently generate an arbitrarily large amount of independent samples $\tilde { \mathbf { u } } _ { \mu _ { 0 } , 1 } , \hdots , \tilde { \mathbf { u } } _ { \mu _ { 0 } , J }$ by sampling the latent variables $z _ { 1 } , . . . , z _ { J } \sim { \mathcal { N } } ( 0 , \mathbf { I } _ { n } )$ , computing the latent coeficients $\phi ( \mu _ { 0 } , z _ { j } )$ and expanding them over the POD basis, $\tilde { \mathbf { u } } _ { \mu _ { 0 } , j } : = \mathbf { V } \phi ( \mu _ { 0 } , z _ { j } )$ . The generated ensemble then allows for a complete characterization of the uncertainties and a proper description of the conditional law of $\mathbf { u } _ { \mu _ { 0 } }$

Algorithm 1: Ofline construction of the POD-CNF surrogate   
Input: Sampling density $\xi ,$ total number of snapshots $S ,$ number of validation samples $N _ { \mathrm { v a l i d } }$ , number of test   
samples $N _ { \mathrm { t e s t } }$ , POD tolerance $\delta ,$ trainable CNF architectures $\{ \phi _ { \pmb { \eta } } ^ { ( j ) } \} _ { j = 1 } ^ { r }$ , optimization routines   
$\{ \mathcal { R } _ { j } \} _ { j = 1 } ^ { r } .$   
Output: POD-CNF Φ   
Step 1. Data collection   
$[ \pmb { \mu } _ { 1 } , \dots , \pmb { \mu } _ { S } ] $ sample from $\xi$   
for $i = 1 , \ldots , S$ do   
u<sub>i</sub> ← sample random event $\theta _ { i }$ and compute $\mathbf { u } _ { \mu _ { i } } ( \pmb \theta _ { i } )$   
end   
$N  S - ( N _ { \mathrm { v a l i d } } + N _ { \mathrm { t e s t } } )$   
for $i = 1 , \ldots , N _ { \mathrm { v a l i d } }$ set $\mathbf { u } _ { i } ^ { \mathrm { v a l i d } }  \mathbf { u } _ { N + i }$   
for $i = 1 , \dots , N _ { \mathrm { t e s t } }$ set ${ \bf u } _ { i } ^ { \mathrm { t e s t } } { \bf \Sigma }  { \bf u } _ { N + N _ { \mathrm { v a l i d } } + i }$   
Step 2. POD basis extraction   
$\mathbf { S } \gets [ \mathbf { u } _ { 1 } , \dots , \mathbf { u } _ { N } ]$ assemble snapshots matrix   
U, Σ, W ← compute SVD so that $\mathbf { S } = \mathbf { U } \pmb { \Sigma } \mathbf { W } ^ { \top }$   
$\sigma _ { 1 } , . . . , \sigma _ { \operatorname* { m i n } \{ N , N _ { h } \} } \gets \mathrm { d i a g } ( { \Sigma } )$ extract singular values in decreasing order   
$\begin{array} { r } { n  2 \operatorname* { m i n } \{ m : [ \sum _ { i > 2 m } \sigma _ { i } ^ { 2 } ] / [ \sum _ { i = 1 } ^ { \operatorname* { m i n } \{ N _ { h } , N \} } \sigma _ { i } ^ { 2 } ] \leq \delta \} } \end{array}$ select even POD dimension   
$\mathbf { V }  \mathbf { U } [ : , 1 : n ]$ extract POD basis via truncation   
Step 3. Train and select CNF module   
for $j = 1 , \dots , r$ do   
$\eta _ { * } ^ { ( j ) } $ minimize NL $\begin{array} { r } { \mathbf { \eta } _ { - } ^ { ( j ) } ( \pmb { \eta } ) = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } } \end{array}$ log $\hat { f } _ { \eta , \mu _ { i } } ^ { ( j ) } ( \mathbf { V } ^ { \top } \mathbf { u } _ { i } )$ numerically using optimization routine ${ \mathcal { R } } _ { j }$   
end   
$\begin{array} { r } { j _ { * } \gets \arg \operatorname* { m i n } _ { j \in \{ 1 , \dots , r \} } - \frac { 1 } { N _ { \mathrm { v a l i d } } } \sum _ { i = 1 } ^ { N _ { \mathrm { v a l i d } } } \log \hat { f } _ { \eta _ { * } } ^ { ( j ) } \left( \mathbf { V } ^ { \top } \mathbf { u } _ { i } ^ { \mathrm { v a l i d } } \right) } \end{array}$ , select best-in-validation CNF module   
Assemble the trained $P O D \ – C N F$   
$\Phi  \mathbf { V } \phi _ { \eta _ { * } } ^ { ( j _ { * } ) }$ , optionally print test NLL   
return Φ

The accuracy of the model can be certified by computing suitable performance metrics over the test set, such as the NLL in the latent space and the POD projection error. Estimating the AWD in (7) is also an option, but this generally only possible in ideal scenarios. In order to approximate (7) numerically, in fact, one needs access to a test set with a nested structure, namely $\{ \mu _ { i } ^ { \mathrm { t e s t } } , \mathbf { u } _ { \pmb { \mu } _ { i } ^ { \mathrm { t e s t } } } ( \pmb { \theta } _ { i , j } ) \}$ with $j = 1 , \dots , N _ { \mathrm { t e s t } }$ and $i = 1 , \ldots , \ell .$ While this is generally not always possible, in our experiments we will force this assumption, efectively overwriting our previous definition of "test set". This will be essential to ensure a rigorous and comprehensive assessment of the probabilistic surrogates, which we find particularly relevant given the exploratory and pioneering nature of the present work.

## 4 Numerical experiments

In this Section, we assess the capabilities of the POD-CNF framework in a controlled simulation setting. Specifically, we investigate whether the proposed approach can approximate the solution of a parametric problem whilst capturing additional stochasticities induced by suitable hidden parameters. This setting thus serves as a proof of concept for the proposed methodology, allowing us to assess its ability to recover the relevant stochastic structures in a well-defined environment. As such, the present analysis represents a preliminary yet necessary step towards more challenging scenarios, including applications to real experimental data, which are beyond the scope of this work.

We consider two benchmark problems: an advection-difusion process in a partially controlled Stokes flow (Section 4.2) and a linear elasticity problem involving a biphasic material with a stochastic interface (Section 4.3). The two test cases serve to demonstrate the capabilities of POD-CNF in both scalar and vector-valued settings. In both cases, the goal is to approximate the conditional distribution of the solution field given deterministic parameters, µ, whilst accounting for the uncertainty induced by unobserved hidden variables, θ. The analysis is based on diferent performance metrics and benchmarks which are briefly outlined in Section 4.1.

We anticipate that, while this Section focuses on assessing the generative capabilities of POD-CNF, a preliminary investigation of its application to uncertainty quantification, including forward and inverse problems, is discussed immediately after: see Section 5.

## 4.1 Numerical setup

Performance metrics Let $\mathbf { u } _ { \mu }$ denote the stochastic ground truth solution, conditional on $\pmb { \mu } \in \mathcal { P }$ . Conversely, let $\tilde { \mathbf { u } } _ { \mu }$ denote the output of a given (trained) probabilistic surrogate.

As a first metric, we consider the reduced negative-loglikelihood over the test set, computed as

$$
\mathsf { N L L } _ { \mathrm { t e s t } } : = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \hat { f } _ { \pmb { \mu } _ { i } ^ { \mathrm { t e s t } } } \big ( \mathbf { V } ^ { \top } \mathbf { u } _ { i } ^ { \mathrm { t e s t } } \big )\tag{23}
$$

where $\hat { f } _ { \mu , { } }$ is the density of the reduced random vector $\mathbf { V } ^ { \top } \tilde { \mathbf { u } } _ { \mu } .$ , whereas each $\mathbf { u } _ { i } ^ { \mathrm { t e s t } }$ is a single independent realization of $\mathbf { u } _ { \mu _ { i } ^ { \mathrm { t e s t } } }$ . As discussed in 3.2, this metric provides a global indication of the goodness of fit; furthermore, it has the advantage of requiring a single FOM evaluation for each $\mu _ { i } ^ { \mathrm { t e s t } }$ . In general, this metric is specifically designed for POD-based probabilistic surrogates that admit an absolutely continuous density at the latent level.

As a second set of metrics, we consider more general indicators based on approximated Wasserstein distances. In general, these require repeated conditional evaluations of both the FOM and the surrogate and are thus much more expensive to evaluate. The idea goes as follows. Given any $\mu _ { 0 } \in \mathcal { P }$ , we exploit the FOM and the surrogate to generate $m _ { \mathrm { g e n } }$ independent realizations of both models, $\mathbf { u } _ { \mu _ { 0 } } ^ { ( 1 ) } , \dots , \mathbf { u } _ { \mu _ { 0 } } ^ { ( \dot { m } _ { \mathrm { g e n } } ) }$ and $\tilde { \mathbf { u } } _ { \mu _ { 0 } } ^ { ( 1 ) } , \dots , \tilde { \mathbf { u } } _ { \mu _ { 0 } } ^ { ( m _ { \mathrm { g e n } } ) }$ , respectively. The two clouds of points in $\mathbb { R } ^ { N _ { h } }$ are then compared to each other using a discrete version of the Wasserstein distance $\tilde { \mathcal { W } } _ { 1 }$ , defined as

$$
\begin{array} { r l } & { \tilde { \mathcal { W } } _ { 1 } ( \{ \mathbf { u } _ { \mu _ { 0 } } ^ { ( 1 ) } , \dots , \mathbf { u } _ { \mu _ { 0 } } ^ { ( m _ { \mathrm { g e n } } ) } \} , \ \{ \bar { \mathbf { u } } _ { \mu _ { 0 } } ^ { ( 1 ) } , \dots , \bar { \mathbf { u } } _ { \mu _ { 0 } } ^ { ( m _ { \mathrm { g e n } } ) } \} ) = } \\ & { \qquad = \operatorname* { m i n } \left\{ \displaystyle \sum _ { i , j } \Gamma _ { \{ i , j \} } \| \mathbf { u } _ { \mu _ { 0 } } ^ { ( i ) } - \mathbf { u } _ { \mu _ { 0 } } ^ { ( j ) } \| \mathrm { ~ w h e r e ~ } \Gamma \in \mathbb { R } ^ { m \times q } \ : \ \sum _ { j } \Gamma _ { \{ i , j \} } = \sum _ { i } \Gamma _ { \{ i , j \} } = 1 / m _ { \mathrm { g e n } } \ \forall i _ { 0 } , j _ { 0 } \right\} . } \end{array}\tag{24}
$$

The above is used as proxy for the true Wasserstein distance between the FOM and generated distributions, that is

$$
\mathscr { W } _ { 1 } ( \mathbf { u } _ { \mu _ { 0 } } , \tilde { \mathbf { u } } _ { \mu _ { 0 } } ) \approx \tilde { \mathscr { W } } _ { 1 } ( \{ \mathbf { u } _ { \mu _ { 0 } } ^ { ( 1 ) } , \dots , \mathbf { u } _ { \mu _ { 0 } } ^ { ( m _ { \mathrm { g e n } } ) } \} , \ \{ \tilde { \mathbf { u } } _ { \mu _ { 0 } } ^ { ( 1 ) } , \dots , \tilde { \mathbf { u } } _ { \mu _ { 0 } } ^ { ( m _ { \mathrm { g e n } } ) } \} ) .\tag{25}
$$

In our implementation, we compute (24) using the pot Python library, with a computational overload of $O ( m _ { \mathrm { g e n } } ^ { 3 } )$ excluding generation costs.

This local analysis is exploited in two ways. First, to us to assess the performances of the model over a specific parametric scenario $\mu _ { 0 } \in \mathcal { P }$ . Second, to investigate the accuracy of the model throughout $\mathcal { P } ;$ this is done by computing (24) for multiple, independent, parametric configurations, randomly drawn from $\mathcal { P }$ . We generally set $m _ { \mathrm { g e n } } = 1 0 0 0$ in the first case and $m _ { \mathrm { g e n } } = 2 0 0$ in the second case to mitigate the computational overload.

Benchmarks We consider two benchmarks against which we compare our proposed strategy, in two diferent test cases. The first benchmark represents an idealized comparison with purely deterministic surrogates. Rather than implementing and evaluating individual deterministic methods, this analysis is carried out by considering an ideal reference predictor, which serves as a baseline for all deterministic models. To this end, we notice that for any $\mu _ { 0 } \in \mathcal { P }$ all deterministic surrogates must return a single, nonaleatoric prediction in $\mathbb { R } ^ { N _ { h } }$ . In mean-square terms, it is wellknown that the optimal such prediction is given by the conditional expectation $\mathbb { E } [ { \mathbf { u } } _ { \mu } \mid \mu = \mu _ { 0 } ] = \mathbb { E } [ { \mathbf { u } } _ { \mu _ { 0 } } ]$ , see, e.g., [54, Chapter 9]. We therefore take $\mu \mapsto \mathbb { E } [ \mathbf { u } _ { \mu } ]$ as our ideal deterministic baseline: no deterministic surrogate can improve upon its mean-square accuracy, regardless of its architecture or training procedure. In practice, for any given $\mu _ { 0 } \in \mathcal { P }$ we approximate the ideal prediction by the empirical mean of a FOM-generated population, $\begin{array} { r } { \overline { { \mathbf { u } } } _ { \mu _ { 0 } } : = m _ { \mathrm { g e n } } ^ { - 1 } \sum _ { i = 1 } ^ { m _ { \mathrm { g e n } } } \mathbf { u } _ { \mu _ { 0 } } ^ { ( i ) } } \end{array}$ From a probabilistic perspective, because the output of a deterministic surrogate is intrinsically degenerate, its accuracy cannot be assessed reliable using the negative log-likehood. For this reason, when analyzing the deterministic baseline, we focus on Wasserstein-based metrics only. In this case (25) becomes

$$
\tilde { \mathcal { W } } _ { 1 } ( \{ \mathbf { u } _ { \mu _ { 0 } } ^ { ( i ) } \} _ { i = 1 } ^ { m _ { \mathrm { g e n } } } , \{ \mathbf { \overline { { u } } } _ { \mu _ { 0 } } \} ) = \frac { 1 } { m _ { \mathrm { g e n } } } \sum _ { i = 1 } ^ { m _ { \mathrm { g e n } } } \| \mathbf { u } _ { \mu _ { 0 } } ^ { ( i ) } - \mathbf { \overline { { u } } } _ { \mu _ { 0 } } \| \approx \mathbb { E } \| \mathbf { u } _ { \mu _ { 0 } } - \mathbb { E } [ \mathbf { u } _ { \mu _ { 0 } } ] \| = \mathcal { W } _ { 1 } ( \mathbf { u } _ { \mu _ { 0 } } , \mathbb { E } [ \mathbf { u } _ { \mu _ { 0 } } ] ) .\tag{26}
$$

As a second benchmark, instead, we consider POD-GP [24]. The latter is an established approach for probabilistic reduced-order modeling that is based on the integration of POD and Gaussian Process (GP) regression. From a generative standpoint, POD-GP can be seen as a model of the form

$$
\Phi _ { \mathrm { P O D - G P } } ( \pmb { \mu } , z ) = \sum _ { j = 1 } ^ { n } \left[ m _ { j } ( \pmb { \mu } ) + z _ { j } s _ { j } ( \pmb { \mu } ) \right] \mathbf { v } _ { j } ,\tag{27}
$$

where $z _ { j }$ and $\mathbf { v } _ { j }$ are the jth component and the jth column of $z \sim \mathcal { N } ( 0 , \mathbf { I } _ { n } )$ and V, respectively, whereas $m _ { j }$ and $\sigma _ { j }$ are the predicted mean and standard deviation returned by the jth GP. In the POD-GP approach, in fact, a sequence of n independent GPs is fitted to approximate the parameter-to-reduced state map, where each $\mathrm { G P }$ focuses on a diferent POD coeficient. In our experiments, we fit POD-GP by leveraging the same data used to design and train POD-CNF. Specifically, when implementing POD-GP, we exploit the validation set to estimate the hyperparameters of each GP, and use the training set to fit the models and obtain the final surrogate. All GPs are implemented using a squaredexponential (RBF) covariance kernel for the prior, scaled by a trainable constant, together with an additive white-noise kernel. The signal variance, length scale, and noise variance are optimized in the ranges $[ 1 0 ^ { - 6 } , 1 0 ^ { 2 } ] , [ 1 0 ^ { - 2 } , 1 0 ^ { 1 } ]$ , and $[ 1 0 ^ { - 8 } , 1 0 ^ { 0 } ]$ , respectively. To ensure that such ranges are meaningful, all inputs and outputs are suitably normalized before training.

An important feature of POD-GP is that its predictive distributions are always Gaussian. Precisely, if we denote by $\mathbf { m } _ { \mathrm { G P } } ( \pmb { \mu } ) : = [ m _ { 1 } ( \pmb { \mu } ) , \dots m _ { n } ( \pmb { \mu } ) ]$ and $\Sigma _ { \mathrm { G P } } ( \pmb { \mu } ) : = \mathrm { d i a g } \left( s _ { 1 } ( \pmb { \mu } ) , \dots , s _ { n } ( \pmb { \mu } ) \right)$ , then Φ<sub>POD-GP</sub> $\mathbf { \Lambda } ( \pmb { \mu } , \cdot ) \sim \mathcal { N } ( \mathbf { V } \mathbf { m } ( \pmb { \mu } ) , \mathbf { V } \pmb { \Sigma } ( \pmb { \mu } ) \mathbf { V } ^ { \top } )$ . Similarly, at the latent level, $f _ { \mu }$ coincides with the density of a multivariate normal $\mathcal { N } ( \mathbf { m } ( \pmb { \mu } ) , \pmb { \Sigma } ( \pmb { \mu } ) )$

## 4.2 Advection-Difusion in a Stokes Flow

As a first test case, we consider the passive transport of a chemical in a rectangular chamber equipped with three rotating cylinders, cf. Fig. 1. Here, $\Omega = \Omega _ { 0 } \setminus ( \mathcal { C } _ { 1 } \cup \mathcal { C } _ { 2 } \cup \mathcal { C } _ { 3 } )$ with $\Omega _ { 0 } = ( 0 , 3 ) \times ( 0 , 2 )$ . This setup simulates a physical mixer where the cylinders rotate at specific velocities, driving the advection field. The objective is to construct a surrogate that can predict the concentration field of the chemical, $u = u ( { \pmb x } )$ , given the configuration of the mixer, while simultaneously accounting for potential uncertainties.

To model the ground truth mechanism, and produce the desired variability, we use the following sets of parameters.

• Conditioning Parameters $\pmb { \mu } = [ c _ { 1 } , c _ { 2 } , c _ { 3 } ]$ . Here, $c _ { i }$ is the rotational velocity of the ith cylinder. Such parameters encode what is known about the configuration of the system, serving as controllable inputs for the mixer. We allow each $c _ { i }$ to vary in a pre-defined range, $c _ { i } \in [ - 5 , 5 ]$

• Hidden Parameters $\theta = [ \kappa , \alpha ]$ . We use κ to model the difusion coeficient of the chemical, and α for the inflow angle of the fluid. We exploit such parameters to introduce an additional, formally unknown, source of variability, which we obtain by letting $\kappa \sim \mathcal { U } [ 0 . 0 0 2 5 , 0 . 0 0 5 ]$ and $\alpha \sim \mathcal { U } [ - \pi / 4 , \pi / 4 ]$ . The first hidden parameter accounts for the fact that, although the identity of the difusing chemical is known, its efective transport properties may vary across experiments as a result of changes in, e.g., temperature, solvent composition, or other experimental conditions that can alter key properties such as the molecular difusivity. These alternations are assumed to be out-of-control, making the efective value of κ formally unknown form the outside. Similarly, the stochasticity in the inflow condition can be though as a way to model imperfections in the injection mechanism or other perturbations taking place outside of the mixing chamber.

Due to the stochastic nature of κ and $\alpha ,$ a fixed configuration of cylinder velocities $\pmb { \mu }$ does not yield a unique solution but rather a distribution of physically consistent concentration fields $\mathsf { p } ( u | \pmb { \mu } ) \mathrm { : }$ : see, e.g., Fig. 2.

Mathematical Formulation and Discretization As anticipated, we model the concentration field $u ( x ; \mu , \theta )$ as the solution to the following advection-difusion equation,

$$
\left\{ \begin{array} { l l } { - \kappa \Delta u ( x ; \pmb { \mu } , \pmb { \theta } ) + \mathbf { b } ( x ; \pmb { \mu } , \pmb { \theta } ) \cdot \nabla u ( x ; \pmb { \mu } , \pmb { \theta } ) = 0 } & { \mathrm { i n ~ } \Omega , } \\ { \nabla u ( x ; \pmb { \mu } , \pmb { \theta } ) \cdot \pmb { n } = 0 } & { \mathrm { o n ~ } \partial \Omega \setminus \Gamma _ { i n } , } \\ { u ( x ; \pmb { \mu } , \pmb { \theta } ) = g ( \pmb { x } ) } & { \mathrm { o n ~ } \Gamma _ { i n } , } \end{array} \right.\tag{28}
$$

where $\Gamma _ { i n } = \{ 0 \} \times [ 0 , 2 ]$ , n is the external unit normal, $g ( \pmb { x } ) = g ( x _ { 1 } , x _ { 2 } ) = \exp { ( - 2 5 ( x _ { 2 } - 1 ) ^ { 2 } ) }$ is a fixed boundary condition, while b is the underlying velocity field. The latter, which depends on $\alpha , c _ { 1 } , c _ { 2 } , c _ { 3 }$ , is determined by the stationary Stokes equations for an incompressible viscous fluid, neglecting inertial terms:

$$
\left\{ \begin{array} { l l } { - \Delta \mathbf { b } ( x ; \pmb { \mu } , \pmb { \theta } ) + \nabla p ( \pmb { x } ; \pmb { \mu } , \pmb { \theta } ) = \mathbf { 0 } } & { \mathrm { i n ~ } \Omega , } \\ { \nabla \cdot \mathbf { b } ( x ; \pmb { \mu } , \pmb { \theta } ) = 0 } & { \mathrm { i n ~ } \Omega , } \end{array} \right.\tag{29}
$$

![](images/048a8af45e55daf2cf07bbc7063a78e3751c93336ec76e39ab8dcc457b5d1bc2.jpg)

![](images/1bc57d456849f308dac159a14d8775d45ba8eaf5baeafa6309a83cafba014678.jpg)

Figure 1. Problem domain for the advection-difusion problem (left) and corresponding discretization (right).  
![](images/5468b4afb813e0619f86666fbe78420a34c9eec3dce4558f2750efb750b1092c.jpg)  
Figure 2. Advection difusion problem. Two random solutions associated with the same operating conditions $\pmb { \mu } = [ - 1 , 2 , - 5 ]$ The variability is induced by the hidden parameters, θ, whose values are assumed to be unknown (to the user, and to the surrogate).

where $p$ represents the fluid pressure. The above is complemented by the following boundary conditions,

$$
\left\{ \begin{array} { l l } { \mathbf { b } ( x ; \mu , \theta ) = [ \cos \alpha , \sin \alpha ] ^ { \top } g ( x ) } & { \mathrm { o n ~ } \Gamma _ { i n } , } \\ { \mathbf { b } ( x ; \mu , \theta ) = c _ { i } [ - ( x _ { 2 } - x _ { 2 } ^ { i } ) , x _ { 1 } - x _ { 1 } ^ { i } ] ^ { \top } } & { \mathrm { o n ~ } \partial \mathcal { C } _ { i } , \quad i \in \{ 1 , 2 , 3 \} , } \\ { \mathbf { b } ( x ; \mu , \theta ) = \mathbf { 0 } } & { \mathrm { o n ~ } \partial \Omega \setminus ( \Gamma _ { i n } \cup \partial \mathcal { C } _ { 1 } \cup \partial \mathcal { C } _ { 2 } \cup \partial \mathcal { C } _ { 3 } ) , } \end{array} \right.\tag{30}
$$

where $( x _ { 1 } ^ { i } , x _ { 2 } ^ { i } )$ is the center of the ith cylinder $\mathcal { C } _ { i }$ . The physics of the problem is deliberately kept simple, as our primary interest lies in addressing the parametric variability of the solution with respect to µ and θ.

As reference solver, we consider a FOM that approximates the solution to (28)–(30) using the Finite Element Method (FEM). To this end, we discretize the spatial domain Ω using an unstructured triangular mesh comprising $N _ { h } = 2 7 6 3$ vertices; see Fig. 1. We use P1 continuous elements for the advection-difusion equation, whereas, in order to satisfy the inf-sup condition, we use the stable Mini-Element pair to solve for the Stokes’ flow equation (continuous P1 elements for the pressure, and P1-bubble for the velocity).

Dataset and POD-CNF configuration For the deterministic parameter, we consider a uniform sampling density over the parameter space $\mathcal { P } = [ - 5 , 5 ] ^ { 3 }$ , so that $\xi \sim \mathcal { U } ( \mathcal { P } )$ . We run the FOM repeatedly in order to produce a total of $S = 6 4 0 0$ independent simulations, which we split into training $( N = 4 8 0 0 )$ , validation $( N _ { \mathrm { v a l } } = 1 2 8 0 )$ , and test $( N _ { \mathrm { t e s t } } = 3 2 0 )$ ) instances. For the POD, we set an energy threshold of $\tau = 1 0 ^ { - 4 }$ . For the CNF module, instead, we select the best-in-validation design among architectures with $K \in \{ 1 2 , 1 6 , 2 4 , 3 2 \}$ alternating afine coupling layers, scaling and translation MLPs of depth $\ell \in \{ 1 , 2 \}$ and width $w \in \{ 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \}$ for each coupling layer, and training routines based on the Adam optimizer [31] with learning rate ${ \sf r } \in \lbrace 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 4 } \rbrace$ , a batch size of 64, early stopping with a patience of 10 epochs, weight decay set to $1 0 ^ { - 5 }$ , and a total of 50 epochs per configuration.

Results I: final design of the POD-CNF surrogate. Following the methodology described in Section 3, we computed the SVD of the training snapshot matrix. The singular value and truncation analysis (Fig. 3) revealed a distinct—but not sharp—energy drop-of. Still, $n = 2 0$ modes eventually sufice to meet the chosen energy threshold. On the validation set, the resulting POD basis V striked a Mean Relative Projection Error (MRPE) of 1.05%,

![](images/fecef26b9500a67f0d0467486f355adf89139bfd89fb0e474dff6aa92d6db3f8.jpg)  
Figure 3. Advection-difusion problem, §4.2. Mean relative reconstruction error (solid blue line) and cumulative energy (dashed black line) versus the number of POD bases n. Truncating at n = 20 guarantees a projection error in euclidean norm equal to 1.05%.

$$
\mathsf { M R P E } _ { \mathrm { v a l i d } } : = \frac { 1 } { N _ { \mathrm { v a l i d } } } \sum _ { i = 1 } ^ { N _ { \mathrm { v a l i d } } } \left[ \frac { \lVert \mathbf { u } _ { i } ^ { \mathrm { v a l i d } } - \mathbf { V } \mathbf { V } ^ { \top } \mathbf { u } _ { i } ^ { \mathrm { v a l i d } } \rVert } { \lVert \mathbf { u } _ { i } ^ { \mathrm { v a l i d } } \rVert } \right] ,\tag{31}
$$

confirming the accuracy of the reduced representation.

The CNF was thus configured to operate in a 20-dimensional space under the conditioning of the physical parameters $\pmb { \mu } \in \mathbb { R } ^ { 3 }$ . Following the results of hyperparameter tuning procedure, a CNF architecture with $K = 2 4$ coupling layers, and internal $\mathrm { M L P }$ units of depth $\ell = 2$ and width $w = 1 2 8$ , was selected. The latter was then trained for 100 epochs using the Adam optimizer with $\mathsf { { r } = 1 0 ^ { - 3 } }$ and a weight decay of $1 0 ^ { - 5 }$ , to obtain the final model. The ofline training procedure proved to be highly eficient, requiring approximately 10 minutes on a single NVIDIA T4 GPU, cf. Table 1.

Results II: model performance on a selected $\mu _ { 0 } \in \mathcal { P }$ . Evaluating the performance of a probabilistic surrogate is, by itself, a fairly complex task. For this reason, we start by assessing the accuracy of the trained POD-CNF on a representative test configuration, obtained by fixing the values of the physical parameters to $\pmb { \mu } _ { 0 } : = [ - 1 , 4 , - 5 ]$ . In particular, we aim at understanding if the generative model is actually able to capture the conditional law of u ${ \bf { \lvert } } \mu = \mu _ { 0 }$

Fig. 4a compares a random solution obtained using the FOM, with $\pmb { \mu } = \pmb { \mu } _ { 0 }$ and $\pmb { \theta } \sim \mathcal { U } [ 0 . 0 0 2 5 , 0 . 0 0 5 ] \otimes \mathcal { U } [ - \pi / 4 , \pi / 4 ]$ drawn at random, against three i.i.d. realizations sampled from the trained POD-CNF model. As we can see, all generated solutions exhibit advection patterns that are consistent with the selected configuration of the system (recall that $\pmb { \mu } _ { 0 }$ defines the speed of the rotating cylinders). The variability in the outputs of the POD-CNF shows how the latter is efectively attempting at capturing an intrinsic stochasticity that is not imputable to $\pmb { \mu }$ alone. Most importantly, such variability is not uncontrolled, but rather presents a precise structure that makes the predicted outputs efectively compatible with the FOM simulation.

Further evidence of this fact can be found in Fig. 5. There, we compare the POD-CNF log-likelihood (in reduced space) of diferent FOM simulations conditionally on $\pmb { \mu } _ { 0 } .$ . Each of these simulations has been extracted from the test set and corresponds to a diferent $\mu \in \mathcal P$ . The reduced log-likelihood shows a clear trend, with FOM simulations becoming less and less likely as their physical configuration, $\textstyle \mu ,$ departs from the one assumed by the POD-CNF, $\pmb { \mu } _ { 0 } .$ This confirms that the proposed surrogate is able to discriminate in between diferent scenarios, efectively assigning a higher likelihood to those physical solutions that are consistent with the prescribed input condition.

To corroborate these qualitative findings, a complementary statistical analysis, comparing true and predicted dis tributions, has been carried out. To this end, we sampled $m _ { \mathrm { g e n } } = 1 0 0 0$ new simulations using both the FOM and the POD-CNF, all conditional on $\pmb { \mu } = \pmb { \mu } _ { 0 }$ . The two populations have then been compared using the (discrete) Wasserstein distance $\tilde { \mathcal { W } } _ { 1 }$ , computed as a proxy for $W _ { 1 } ( { \bf u } _ { \pmb { \mu } _ { 0 } } , { \bf V } \phi ( { \pmb { \mu } _ { 0 } } ) ) ;$ cf. Section 4.1. Results are in Table 1. Notably, POD-CNF scores particularly well, with an estimated Wasserstein distance of 0.6735. In particular, the method results 56% more accurate that any deterministic surrogate potentially conceivable, demonstrating that its capabilities extend far beyond the standard regression to the mean. In contrast, when repeating the same analysis for POD-GP, we note a significant drop in accuracy. This latter can fact can be explained in at least two ways. The first one, is that the current analysis is

(a)  
![](images/2971edae4fde5d00d73668532b338cf7fec97f0b6b9eb2562bf73eed758d0f95.jpg)

![](images/65144182e1bc2f838eb35ebadd413be1c00b49f3490dd8bd0890c91bcb345b92.jpg)  
µ=[−1,4, − 5]

![](images/04979f1d6a6c5432546084289ea54aa475501763d6d11fd336216830954a445b.jpg)

![](images/c943f75ed64a21f25425068ff2a967aa20c6eb1e8f597da741e4962260300857.jpg)

-1.0 -0.8 0.6 0.4 -0.2 0.0 -1.0 -0.8 -0.6 -0.4 -0.2 0.0

(b)  
![](images/86a28a4f39e793814e88bf5f49f4b0fb7a3893ec0e34ba025faced24b6c8cd78.jpg)

![](images/f7283d949be34b584a25a2499963482fc6f046dc1102f080f8dbfe569566514e.jpg)

![](images/14ae8db31b304f082f9dd688637e2e179044c1a98630cd77da5ec5597cb5779d.jpg)  
μ=[5,5,5]

![](images/5288aef72da90d2f95d262f0a5c56e6c8372a4928d4a7de4a550aa8832c220d3.jpg)  
(c)

![](images/9dc0de6f8038ff29e7a5aff0621b777d63216c46d7afed5143fedd186d96fad3.jpg)

![](images/6f8fc9beb4f402134fe4c59e9e03844f3d8404b1b7312fb8fa702a1bb4b215da.jpg)  
µ=[−5,0,−5]

![](images/125c3ae3fae38b189ffd6bddf207e1ac3ca336da9b2cd7a50a5595fb78dffc31.jpg)

![](images/1882b4f2ed33114ae222244ac879aa42761e46e079cb04f7dc906ca4899c7392.jpg)  
Figure 4. Advection-difusion problem, §4.2. Solutions fields across diferent parametric configurations: (a) $\pmb { \mu } _ { 0 } = [ - 1 , 4 , - 5 ]$ (b) $\mu _ { 1 } = [ 5 , 5 , 5 ]$ , and (c) $\pmb { \mu } _ { 2 } = [ - 5 , 0 , - 5 ]$ . For each row, the leftmost panel shows a ground-truth snapshot, computed with the numerical solver, while the three panels to its right show independent realizations sampled from the POD-CNF surrogate model. Configuration (a) corresponds to the parameter $\mu _ { 0 } \in \mathcal { P }$ chosen for the local analysis.

entirely focused on a limit scenario, as the parametric configuration under study falls at the boundary of the parameter space, $\mu _ { 0 } \in \partial \mathcal { P }$ . As such, even with the prescribed training data, the GP remains highly uncertain about its prediction: this results in an inflation of the conditional variance $\mathrm { V a r } ( { \bf u } | \mu _ { 0 } )$ , and a consequent degradation in the approximation of the conditional law (see also Fig. 6). The second explanation concerns the fact that the predictive distribution of a GP is necessarily Gaussian. In our setting, where stochasticity arises from hidden variables rather than from external white noise, this simplification ends up posing major limitations.

Our conclusion are further supported by the results shown in Fig. 6. By focusing on eight monitoring points spread across the spatial domain, the latter compares the marginal distributions of the true FOM population against the ones obtained with POD-CNF and POD-GP, respectively. Because 1-dimensional distributions are much easier to inspect compared to high-dimensional ones, this allows us to make both quantitative and qualitative considerations. For instance, we can see how POD-CNF correctly identifies regions with higher volatility, matching both the range and distribution shape of the values attained by the solution field (point P1). Highly peaked distributions (points P5–P6) are also accurately represented, even when they exhibit asymmetry. This capability is particularly valuable if we consider that conventional probabilistic surrogates, such as POD-GP, generally fail at reproducing these features. Clearly, the model also shows some limitations in certain points of the domain (P7-P8), although the performances remain satisfactory.

Results III: global performance. To rigorously evaluate the generalization capabilities of POD-CNF, the previous analysis has been extended to 50 unseen parameter values, randomly drawn from the test set. In this case, in order to mitigate the computational overload, the discrepancy in $\mathcal { W } _ { 1 }$ metric has been estimated using only $m _ { \mathrm { g e n } } = 2 0 0$ samples per parametric configuration. 7 shows the distribution of such errors across the 50 configurations under study for POD-CNF, POD-GP and the deterministic baselines. Averages and standard deviations are reported in Table 1. Notably, POD-CNF consistently outperforms its competitors, demonstrating remarkable generalization capabilities across all parametric regimes. The same conclusion can be drawn by looking at the negative log-likelihood, Table 1 (note: being computationally cheaper to evaluate, the latter as been computed on the whole test set). Finally, Fig. 4a-c shows how POD-CNF is able to produce meaningful solutions conditionally on the desired parametric configuration. Of note, we can see that the POD-CNF generated samples show a clear variability in the inflow condition at the left boundary. This is particularly interesting if we consider that our model was never given information about the existence of an inflow angle α. In this sense, we can see how POD-CNF is efectively capturing the stochasticity produced by the latent, hidden, parameters. This is fundamentally diferent from what POD-GP normally does. The latter, in fact, is built based on a Bayesian paradigm, which makes the model sensitive to epistemic uncertainty. While this feature can be particularly valuable in certain applications, here it can obscure the aleatoric component of uncertainty, which is our primary focus.

![](images/3e5c6b70b2ef29fc31066074d3e66608fdb7ab9600092988080f96418dfa8df0.jpg)

Figure 5. Reduced log-likelihood sensitivity of the POD-CNF surrogate evaluated at $\pmb { \mu } _ { 0 }$ against solutions belonging to diferent parameter instances $\pmb { \mu _ { k } }$
<table><tr><td></td><td> $\tilde { \mathcal { W } } _ { 1 }$  at  $\pmb { \mu } _ { 0 }$ </td><td>Average  $\tilde { \mathcal { W } } _ { 1 }$  on  $\mathcal { P }$ </td><td> ${ \mathsf { N L L } } _ { \mathrm { t e s t } }$ </td><td>Training time</td><td>Inference time</td></tr><tr><td>Deterministic baseline</td><td>1.5284</td><td> $1 . 4 3 6 9 \pm 0 . 1 1 0 3$ </td><td></td><td></td><td></td></tr><tr><td>POD-GP</td><td>2.3819</td><td> $1 . 7 9 7 0 \pm 0 . 1 6 0 5$ </td><td> $4 . 5 3 1 5 \pm 0 . 9 1 1 2$ </td><td>54s</td><td>0.29s</td></tr><tr><td>POD-CNF</td><td>0.6735</td><td> $0 . 5 6 9 1 \pm 0 . 0 9 9 7$ </td><td> $- 2 0 . 7 0 5 9 \pm 7 . 6 4 1 6$ </td><td>10m 33s</td><td>0.06s</td></tr></table>

Table 1. Advection-difusion problem, §4.2. Performance metrics and training times for POD-CNF, POD-GP and the deterministic baseline (cf. Section 4.1). First column: surrogate under study. Second column: empirical Wasserstein distance between FOM-generated and surrogate-generated samples $( m _ { \mathrm { g e n } } = 1 0 0 0 )$ . Third column: average Wasserstein distance, computed across 50 random parametric configurations $( m _ { \mathrm { g e n } } = 2 0 0 )$ , and corresponding standard deviation., Fourth column: negative loglikeli hood, averaged across the whole test set (320 joint samples in $\mu , \theta )$ , reported with a 95% confidence interval. Fifth column: fitting time, excluding hyperparameter tuning. Last column: sequential time required to generate a single scenario for a given parametric configuration.

Results IV: sample size sensitivity analysis. We conclude by presenting the results of a complementary analysis, devoted at assessing the sensitivity of the CNF module to the number of training data. To this end, we re-trained the CNF from scratch, keeping the same POD basis $( n = 2 0 )$ and the identified hyperparameters, but progressively reducing the amount of training samples, from 100% (4800 snapshots) down to 5% (240 snapshots). 2 shows the impact of the latter on the performances of POD-CNF.

Interestingly, the model remains quite stable even when halving the amount of training data. This suggests a convergence in the performance: in order to make the surrogate more accurate, instead of adding more data, one would need to either increase the expressivity of the CNF or the number of POD basis. At 25% sample size, the performance starts to deteriorate slightly. However, the most notable drop happens at 10% (and below), where POD-CNF loses most of its value: there, in fact, its performances essentially become on par with the deterministic baseline and POD-GP, cf. Table 1.

![](images/62b4a8603a3650e0e3093950e7dc1230e85aeb2b00892dfb523dacc29326c5e5.jpg)

![](images/bd23cbd6434656ad191d05340d8ce9137a8092b98b7e9915939d06fbf5b3088b.jpg)  
Figure 6. Advection-difusion problem, §4.2. Comparison between FOM and POD-CNF marginal densities (bottom panel) at eight selected monitoring points (top panel).

<table><tr><td>% Data</td><td> $N _ { \mathrm { t r a i n } }$ </td><td> $N _ { \mathrm { v a l } }$ </td><td> $\mathbf { A v e r a g e } \ \tilde { \mathcal { W } } _ { 1 }$   $\mathcal { P }$  on</td><td> $\tilde { \mathcal { W } } _ { 1 }$  at  $\pmb { \mu } _ { 0 }$ </td></tr><tr><td>100%</td><td>4800</td><td>1280</td><td> $0 . 5 6 9 1 \pm 0 . 0 9 9 7$ </td><td>0.6735</td></tr><tr><td>75%</td><td>3600</td><td>960</td><td> $0 . 5 1 0 2 \pm 0 . 0 8 7 2$ </td><td>0.6675</td></tr><tr><td>50%</td><td>2400</td><td>640</td><td> $0 . 5 4 8 7 \pm 0 . 0 9 4 3$ </td><td>0.8187</td></tr><tr><td>25%</td><td>1200</td><td>320</td><td> $0 . 6 6 0 7 \pm 0 . 1 2 8 7$ </td><td>0.9352</td></tr><tr><td>10%</td><td>480</td><td>128</td><td> $1 . 5 0 1 9 \pm 0 . 4 0 2 4$ </td><td>1.8266</td></tr><tr><td>5%</td><td>240</td><td>64</td><td> $2 . 0 2 7 3 \pm 0 . 4 0 2 3$ </td><td>2.3104</td></tr></table>

Table 2. Advection-difusion problem, §4.2. Sensitivity of POD-CNF performance to training dataset size. Evaluation metrics read as in Table 1.

![](images/58d16de20f52cbdb8cb74d36a4fd1c58f941eafe8e12b9195c77b457de7e77db.jpg)  
Figure 7. Advection-difusion problem, §4.2. Global analysis over 50 unseen test configurations. The boxplots show the distribution of the Wasserstein-1 error over the considered test set, comparing FOM and surrogate outputs. From left to right: POD-CNF (red, mean = 0.5673), POD-GP (green, mean = 1.7970) and deterministic baseline (blue, mean = 1.4369).

![](images/dd4a21d247c243b52022fde386876003adee831d449680fe0246b38e604bedfe.jpg)

![](images/eff3c530d7a592b07cf1d6887e554235eb69634c12d6f46b248fc51135effa0d.jpg)  
Figure 8. Linear elasticity problem, §4.3. Physical domain (left) and corresponding discretization (right).

## 4.3 Linear Elasticity

The second numerical experiment extends the proposed framework to a vector-valued setting, addressing the mechani cal response of a square block $\Omega = ( 0 , 1 ) ^ { 2 }$ composed of a biphasic material. The domain features two distinct phases—a "soft" and a "hard" material—separated by a linear interface with an unobserved orientation angle $\theta \ ( \mathrm { F i g . \ 8 } )$ . Deformation is induced by a localized vertical force, acting at the top boundary. The system is governed by the following parameters.

• Conditioning Parameters $\mu = [ m , \delta ]$ . We use these controllable parameters to model the magnitude (m) and the locality (δ) of the vertical force acting on the box. We allow m and δ to vary uniformly, and independently, in the ranges [1, 2] and [0.15, 0.25], respectively, so that $\mathcal { P } : = [ 1 , 2 ] \times [ 0 . 1 5 , 0 . 2 5 ]$

• Latent Parameter θ. We consider a single unobserved parameter, $\theta \in [ 0 , 2 \pi )$ , which we use to prescribe the orientation of the linear interface separating the two material phases in the physical domain, cf Fig. 8. Intuitively, this setup can be thought as a simplified model for an indentation test performed on a heterogeneous, biphasic material block whose internal interface orientation cannot be determined non-destructively prior to loading.

We let $\theta \sim \mathcal { U } ( 0 , 2 \pi )$ Consequently, for a fixed loading configuration $\pmb { \mu } \in \mathcal { P }$ , the displacement field $\pmb { u } = ( u _ { x } , u _ { y } )$ is not unique but rather follows a conditional distribution $\mathsf { p } ( { \pmb u } | { \pmb \mu } )$ dependent on the random realization of the materia interface: see Fig. 9.

μ = [1.000, 0.250]  
![](images/e2f4cfec8feabac2d74903b44cdc3fef0063a1ed5154180e9cd4388ece0e5996.jpg)  
Figure 9. Linear elasticity problem, §4.3. Two random solutions associated with the same operating conditions $\pmb { \mu } = [ 1 . 0 , 0 . 2 5 ]$ The variability is induced by the hidden parameters, $\theta ,$ whose values are assumed to be unknown (to the user, and to the surrogate).

Mathematical Formulation and Discretization. We model the system using the equations of linear static elasticity:

$$
\left\{ \begin{array} { l l } { - \nabla \cdot \left( \lambda _ { \theta } ( \boldsymbol { x } ) ( \nabla \cdot \boldsymbol { u } ) \mathbf { I } + \nu \left( \nabla \boldsymbol { u } + \nabla \mathbf { u } ^ { \top } \right) \right) = - g } & { \mathrm { i n ~ } \Omega , } \\ { \boldsymbol { u } \equiv \mathbf { 0 } } & { \mathrm { o n ~ } \Gamma _ { b o t t o m } , } \\ { \lambda _ { \theta } ( \boldsymbol { x } ) ( \nabla \cdot \boldsymbol { u } ) \boldsymbol { n } + \nu \left( \nabla \mathbf { u } + \nabla \boldsymbol { u } ^ { \top } \right) \boldsymbol { n } \equiv \mathbf { f } _ { m , \delta } ( \boldsymbol { x } ) } & { \mathrm { o n ~ } \Gamma _ { t o p } , } \\ { \lambda _ { \theta } ( \boldsymbol { x } ) ( \nabla \cdot \boldsymbol { u } ) \boldsymbol { n } + \nu \left( \nabla \mathbf { u } + \nabla \boldsymbol { u } ^ { \top } \right) \boldsymbol { n } \equiv \mathbf { 0 } } & { \mathrm { o n ~ } \partial \Omega \setminus \left( \Gamma _ { t o p } \cup \Gamma _ { b o t t o m } \right) . } \end{array} \right.\tag{32}
$$

The first equation represents the static force balance within the domain, where the body force −g accounts for the efect of gravity, and is assumed to be fixed. This equilibrium is subject to mixed boundary conditions. On the lower boundary $\Gamma _ { b o t t o m }$ , a homogeneous Dirichlet condition ${ \pmb u } \equiv { \bf 0 }$ is imposed, meaning that the block is rigidly clamped and experiences no deformation at the bottom. On the upper boundary $\Gamma _ { t o p } .$ , the system is subjected to a localized traction force f exerted by the external loading. This Neumann condition is modeled as a downward vertical force with parametrized magnitude and position, here defined as

$$
\mathbf { f } _ { m , \delta } ( \pmb { x } ) = \mathbf { f } _ { m , \delta } ( x _ { 1 } , x _ { 2 } ) = [ 0 , - m ] ^ { \top } \mathbf { 1 } _ { [ \frac { 1 } { 2 } - \delta , \frac { 1 } { 2 } + \delta ] } ( x _ { 1 } ) .
$$

The remaining lateral boundaries $\partial \Omega \setminus ( \Gamma _ { t o p } \cup \Gamma _ { b o t t o m } )$ , instead, are assumed to be stress-free (homogeneous Neumann condition). The biphasic nature of the material is made explicit in the definition of the Lamé parameters. Indeed, while the shear modulus ν is assumed to be constant across the entire domain, the first Lamé parameter $\lambda _ { \theta }$ is modeled as a piecewise constant function, whose definition depends on the stochastic interface angle θ. More precisely, we let

$$
\lambda _ { \theta } ( \pmb { x } ) = \left\{ \begin{array} { l l } { 0 . 1 \quad \mathrm { i f } \ \pmb { x } \in B _ { \theta } ^ { 1 } , } \\ { 7 . 1 \quad \mathrm { i f } \ \pmb { x } \in \Omega \setminus B _ { \theta } ^ { 1 } , } \end{array} \right.\tag{33}
$$

where $B _ { \theta } ^ { 1 }$ denotes the subdomain of the first material phase, efectively introducing a structural discontinuity within the physical domain.

We discretize the problem using the FEM on a structured triangular mesh (30 × 30 elements - Fig. 8). The displacement field is approximated with continuous piecewise linear functions (P1), while material properties are modeled with piecewise constant functions (P0). The resulting system has $N _ { h } = 1 9 2 2$ degrees of freedom, combining x and y displacements.

We emphasize that, in order favor visual inspections and qualitative assessment, the units considered in Eq. (32) have been magnified on purpose, far beyond the conventional ranges in which the equations of linear elasticity are commonly assumed to provide a reasonable approximation of real-world deformations.

Dataset and POD-CNF configuration. We leverage the FOM to construct a dataset consisting of $N = 6 4 0 0$ independent simulations, partitioned into training $( N _ { \mathrm { t r a i n } } = 4 8 0 0 )$ , validation $( N _ { \mathrm { v a l } } = 1 2 8 0 )$ and test $( N _ { \mathrm { t e s t } } = 3 2 0 )$ sets, following the same split used for the advection-difusion problem. We sample the conditioning parameters uniformly across the parameter space, $\mu \sim \mathcal { U } ( \mathcal { P } )$ , and proceed analogously for the unobserved interface angle, $\theta \sim \mathcal { U } ( 0 , 2 \pi )$ To handle the bivariate nature of the solution field, the $u _ { x }$ and $u _ { y }$ components are flattened and concatenated into a single vector $\mathbf { u } \in \mathbb { R } ^ { N _ { h } }$ . For the POD, instead of fixing an energy threshold a priori, we directly selected the same number of modes used for the previous test case and check their accuracy a posteriori on the validation set. This is to guarantee an easier comparison between the two numerical experiments. Similarly, for the CNF module, we initialize the hyperparameter tuning procedure using the same setup adopted for the advection-difusion problem.

![](images/f8b3b6abf57b64cd5f8e77518b90d7e1d956d679d44a64ac5b69f87289ef6bc1.jpg)  
Figure 10. Linear elasticity problem, §4.3. POD truncation analysis: mean relative reconstruction error on the validation set (blue solid line) and cumulative retained energy (dashed black line) as a function of the number of POD modes n.

Results I: final design of the POD-CNF surrogate. The SVD analysis of the training snapshots is illustrated in Fig. 10. Retiring $n = 2 0$ basis captures 97.15% of the cumulative energy and achieves an average projection error of 0.36% on the validation set, thus yielding the desired accuracy. Following the results of the hyperparameter tuning procedure, instead, a CNF architecture with $K = 1 6$ coupling layers was selected, with each layer featuring conditioner MLPs of depth $\ell = 2$ and width $w = 2 5 6$

Results II: model performance on a selected $\mu _ { 0 } \in \mathcal { P }$ . Similarly to the scalar case, we first focus on a representative test configuration, $\mu _ { 0 } = [ 1 . 5 , 0 . 2 ]$ , falling right at the center of the parameter space.

Fig. 11a compares a random FOM solution with $\pmb { \mu } = \pmb { \mu } _ { 0 }$ and $\theta \sim \mathcal { U } ( 0 , 2 \pi )$ against three i.i.d. realizations sampled from the trained POD-CNF model. The displacements predicted by the surrogate appear consistent with the prescribed operating condition, $\pmb { \mu } _ { 0 }$ , capturing most of the qualitative features observed in the FOM simulation. At the same time, each sample is diferent. In particular, we can see how the model is able to reproduce the variability induced by the interface angle θ, despite not being explicitly aware of it.

Fig. 12 displays the (reduced) log-likelihood, conditioned on the chosen $\pmb { \mu } _ { 0 }$ and evaluated on the test set. The behavior is analogous to the one seen in the advection-difusion problem, with the log-likelihood presenting a clearly decreasing trend in $\| \mu - \mu _ { 0 } \|$ . This further confirms that the stochasticity introduced by POD-CNF is highly structured, allowing the model to distinguish amongst solutions coming from diferent operating conditions.

The results of the quantitative analysis, comparing POD-CNF, POD-GP, deterministic baseline and FOM over $m _ { \mathrm { g e n } } = 1 0 0 0$ generated samples, can be found in Table 3. This time, For the parametric configuration under study, the proposed approach achieves a a Wasserstein-1 error of 0.4561, resulting nearly four times more accurate than POD-GP and improving upon the deterministic baseline by a factor of 83.76%. To verify that the model correctly captures componentwise cross-correlations in the whole physical space, we monitor the 2D joint displacement $( u _ { x } , u _ { u } )$ at four locations across the domain. Fig. 13 compares the resulting joint probability density contours against the FOM reference. Remarkably, POD-CNF is to able to capture several nontrivial features seen in the FOM distribution, including the characteristic multi-modal behavior induced by the interface angle θ. Once again, this is particularly valuable considering that standard Gaussian approximations would generally be unable to replicate such level of complexity.

Results III: global performance. We assess the generalization capabilities of POD-CNF by repeating the previous analysis on 50 unseen loading configurations extracted the test set, this time using $m _ { g e n } = 2 0 0$ samples per instance to reduce computational cost. As shown in Figure 14, the Wasserstein-1 error remains well below the system’s intrinsic variability (deterministic baseline). Across all test cases, POD-CNF achieves an average $\tilde { \mathcal { W } } _ { 1 }$ error of 0.5517, approximately 80% lower than the baseline variability (2.6443). The model also performs significantly better than POD-GP (see also the negative log-likelihood in Table 3), further demonstrating that the two approaches serve entirely diferent purposes, and that POD-CNF is a more natural choice in this framework.

(a)  
![](images/7d021439e525ddc3b5e7b316957b70abc12576d85df2a9cb55263fa2d9a7258c.jpg)

![](images/e70279e94e6eb9ea0ba7faa0b0c57b2a5d0e0015230667755df26a98cce04cb3.jpg)

![](images/24fd8d4dcb3d2e3482d6b66e8ab56f3e62d46ba075b9bbddcbd8c90c69e20102.jpg)

![](images/3407f4c1ae3d0788184dcc656a7f31be0dfa9f434800767f56eee2871d20f961.jpg)

(b)  
![](images/b3cc4fb6361dbff7ff1a2f7a8c712fe88d3bda5a0be384907c5c4bcc26ef5a01.jpg)

![](images/30a42aa7e69d52441de88335d6b04f0a3420c7fb0a3f28ad97c39c535b49121d.jpg)

![](images/d2d9c296dad3cf5e0d3170c3de96e345b87a60fbca9638f3c5db17a91806cef5.jpg)

![](images/8cc4bbc4aea733dbde198d28495d4e54d67e27c1cfdef5888a5816f30f25bd10.jpg)

(c)  
![](images/530317c494260f8ade7624247e0fe1bd4a25a3c7b7ed254f223887b81b6fa13a.jpg)

![](images/a2dee8f89da85461383ffcefceb1615119b5f1fff76935412937641f296c6276.jpg)

![](images/877e62fc7f8141ff58155bfd094e7acc1def442ce2d0448a4ca91a3bf93861d3.jpg)  
µ=[1.75, 0.15]

![](images/936c74333ce2a257f6b23d8fb3c96df5baa2a01db7065905b004d7fd52701a52.jpg)  
Figure 11. Linear elasticity problem, §4.3. Solutions fields across diferent parametric configurations: (a) $\mu _ { 0 } = [ 1 . 5 , 0 . 2 0 ]$ (b) $\mu _ { 1 } = [ 2 . 0 , 0 . 2 5 ]$ , and (c) $\mu _ { 2 } = [ 2 . 0 , 0 . 2 5 ]$ . For each row, the leftmost panel shows a ground-truth snapshot, computed with the numerical solver, while the three panels to its right show independent realizations sampled from the POD-CNF surrogate model. Configuration (a) corresponds to the parameter $\mu _ { 0 } \in \mathcal { P }$ chosen for the local analysis.

![](images/b6e003d24e5e16edcc3faef6249f535882e9367f1205471d32164cbc05335e54.jpg)  
Figure 12. Reduced log-likelihood sensitivity of the POD-CNF surrogate evaluated at $\pmb { \mu } _ { 0 }$ against solutions belonging to diferent parameter instances $\pmb { \mu _ { k } }$

![](images/d69ac040f6cbc03e57159ce1afe02d9e954e8df5d57a62c1006c0aed6a9b6ce7.jpg)

![](images/1080ca4c2ddb715fcea1bf6d77a3aeb170209641b6be2ade0dda5540a6dd7894.jpg)

![](images/bb6977a1507cd9b9be63c98eb1fcce1fc97dbe272f6f0d32e7f8b005f4d4f4dc.jpg)

![](images/5be3265091c438fae105de9003b2b8d7d10ba2382b7f14ff15b88bc198bf22d6.jpg)

![](images/3668de0990b914b28e13f8181a03a87a9c3ead3c6eaef3d609f3c3017a7cf572.jpg)

![](images/4e08632c9598b952db922391c59a897d8878855ce0fa05bf3cc3bb93c81f6878.jpg)  
Figure 13. Linear elasticity problem, §4.3. Top left: computational mesh and locations of the four monitoring points (P1–P4). Bottom and right: joint probability density contours of the horizontal and vertical displacements $( u _ { x } , u _ { y } )$ at P1–P4 for µ<sub>0</sub>, comparing the FOM reference with the POD-CNF generated distribution (left and right of each subplot, respectively).

![](images/7ab52c4b23ff2c58afede4e61fdbed2ca5b401faf79e9b3e197eede97f216cd1.jpg)  
Figure 14. Linear elasticity problem, §4.3. Global analysis over 50 unseen test configurations. The boxplots show the distribution of the Wasserstein-1 error over the considered test set, comparing FOM and surrogate outputs. From left to right: POD-CNF (red, mean = 0.5117), POD-GP (green, mean = 1.8402) and deterministic baseline (blue, mean = 2.6443).

<table><tr><td></td><td> $\tilde { \mathcal { W } } _ { 1 }$  at  $\pmb { \mu } _ { 0 }$ </td><td>Average  $\tilde { \mathcal { W } } _ { 1 }$  on  $\mathcal { P }$ </td><td> ${ \mathsf { N L L } } _ { \mathrm { t e s t } }$ </td><td>Training time</td><td>Inference time</td></tr><tr><td>Deterministic baseline</td><td>2.8098</td><td> $2 . 6 4 4 3 \pm 0 . 3 3 0 8$ </td><td></td><td></td><td></td></tr><tr><td>POD-GP</td><td>1.6707</td><td> $1 . 8 4 0 2 \pm 0 . 3 3 4 3$ </td><td> $- 2 7 . 7 3 8 5 \pm 0 . 4 0 7 6$ </td><td>55.57s</td><td>0.29s</td></tr><tr><td>POD-CNF</td><td>0.4561</td><td> $0 . 5 1 1 7 \pm 0 . 1 2 5 9$ </td><td> $- 1 0 1 . 9 4 7 4 \pm 2 . 3 0 8 2$ </td><td>12m 24s</td><td>0.08s</td></tr></table>

Table 3. Linear elasticity problem, §4.3. Quantitative analysis comparing POD-CNF, POD-GP and the deterministic baseline, cf. §4.1. Table entries read as in Table 1.

<table><tr><td>% Data</td><td> $N _ { \mathrm { t r a i n } }$ </td><td> $N _ { \mathrm { v a l } }$ </td><td>Average  $\tilde { \mathcal { W } } _ { 1 }$   $\mathcal { P }$  on</td><td> $\tilde { \mathcal { W } } _ { 1 }$  at  $\pmb { \mu } _ { 0 }$ </td></tr><tr><td>100%</td><td>4800</td><td>1280</td><td> $0 . 5 1 1 7 \pm 0 . 1 2 5 9$ </td><td>0.4561</td></tr><tr><td>75%</td><td>3600</td><td>960</td><td> $0 . 5 1 1 0 \pm 0 . 1 6 2 4$ </td><td>0.4135</td></tr><tr><td>50%</td><td>2400</td><td>640</td><td> $0 . 5 0 6 2 \pm 0 . 1 3 2 7$ </td><td>0.4936</td></tr><tr><td>25%</td><td>1200</td><td>320</td><td> $0 . 6 7 6 1 \pm 0 . 1 7 9 9$ </td><td>0.6768</td></tr><tr><td>10%</td><td>480</td><td>128</td><td> $1 . 2 9 3 2 \pm 0 . 2 4 0 3$ </td><td>1.4152</td></tr><tr><td>5%</td><td>240</td><td>64</td><td> $2 . 3 9 5 2 \pm 0 . 2 4 9 8$ </td><td>1.3998</td></tr></table>

Table 4. Linear elasticity problem, §4.3. Sensitivity of POD-CNF accuracy to training sample size. Table entries read as in Table 2.

Results IV: sample size sensitivity analysis. We investigate the data eficiency of the architecture by retraining the CNF on progressively smaller datasets, all while keeping the POD basis $( n = 2 0 )$ and the model hyperparameters fixed (Table 4). Consistently with the results seen in the previous test case, the performance remains remarkably stable down to 25% of the original sample size $( N _ { \mathrm { t r a i n } } = 1 2 0 0$ , average $\mathcal { W } _ { 1 } = 0 . 6 7 6 8 )$ . Reducing the training size further leads to a sharp drop in accuracy, suggesting that $N _ { \mathrm { t r a i n } } \approx 1 2 0 0$ is the critical data threshold for this problem.

## 5 Application to Forward and Inverse Uncertainty Quantification (UQ)

The numerical experiments presented in Section 4 exemplify scenarios where the physical model is governed by a PDE subject to intrinsic uncertainties, encoded by latent random parameters. Having assessed the capabilities of POD-CNF in this framework, we now aim at illustrating how the latter can be deployed to tackle Uncertainty Quantification (UQ) tasks. We shall consider both forward and inverse problems, alternating between the two test cases discussed in Section 4.

## 5.1 Forward UQ

Traditionally, Forward UQ aims at characterizing how uncertainties about the configuration of the system propagate to a given Quantity of Interest (QoI). For instance, given a QoI Q that can be expressed as a Lipschitz continuous functional of the state, $\boldsymbol { Q } : \mathbb { R } ^ { N _ { h } } \overset { \cdot } {  } \mathbb { R }$ , a classical problem in Forward UQ concerns the estimation of the conditional expectation

$$
\mathbb { E } [ Q ( { \mathbf { u } } _ { \mu } ) ] = \int Q ( { \mathbf { u } } _ { \mu } ) { \mathsf { p } } ( { \mathbf { u } } | \mu ) d { \mathbf { u } } ,\tag{34}
$$

often approximated via Monte Carlo, or related techniques, as $\begin{array} { r } { \mathbb { E } [ Q ( { \mathbf { u } } _ { \mu } ) ] \approx \sum _ { j = 1 } ^ { N } Q ( { \mathbf { u } } _ { \mu } ^ { ( j ) } ) / N , } \end{array}$ where the ${ \bf u } _ { \mu } ^ { ( j ) } { \bf \Phi } _ { \mathrm { S } }$ are independent FOM simulations, all conditional on $\pmb { \mu } .$ In our setting, we can accelerate this process, by exploiting the generative capabilities of the probabilistic surrogate, efectively bypassing the PDE solver. For instance, a natural

approach is to replace the conventional estimator with

$$
\mathbb { E } [ Q ( \mathbf { u } _ { \pmb { \mu } } ) ] \approx \frac { 1 } { N } \sum _ { j = 1 } ^ { N } Q ( \mathbf { V } \phi ( \pmb { \mu } , z ^ { ( j ) } ) ) .\tag{35}
$$

where $\boldsymbol { z } ^ { ( 1 ) } , \ldots \boldsymbol { z } ^ { ( N ) } \sim \mathcal { N } ( 0 , \mathbf { I } _ { n } )$ are independent Gaussian random vectors. Indeed, if we denote by $\tilde { \mathbf { u } } _ { \mu } : = \mathbf { V } \phi ( \mu , z )$ the output of the POD-CNF, then, as a direct consequence of the Kantorovich-Rubinstein duality,

$$
| \mathbb { E } [ Q ( { \mathbf { u } } _ { \mu } ) ] - \mathbb { E } [ Q ( \tilde { \mathbf { u } } _ { \mu } ) ] | \leq \mathrm { L i p } ( Q ) \mathcal { W } _ { 1 } ( \mathbf { u } _ { \mu } , \tilde { \mathbf { u } } _ { \mu } ) ,\tag{36}
$$

where $\operatorname { L i p } ( Q )$ is the Lipschitz constant of the $\mathrm { Q o I } ,$ and the $\mathcal { W } _ { 1 }$ -term can be bounded as discussed in Section 3.1. Because the right-hand side of (35) is nothing but the Monte Carlo approximation of $\mathbb { E } \big [ Q ( \tilde { \mathbf { u } } _ { \mu } ) \big ]$ ], this readily motivate the definition of the proposed estimator.

To demonstrate how the idea unfolds in practice, we present a simple application featuring the advection-difusion problem discussed in Section 4.2. We consider five diferent QoIs, defined as in Table 5: each of them relates to a diferent feature of the solution field. For instance, $Q _ { 3 }$ quantifies the amount of solute accumulated around the bottom-right cylinder, in comparison with the other two. Conversely, $Q _ { 5 }$ identifies potential concentration spikes in the wake region of the mixer.

Figure 15 shows the results obtained for the diferent QoIs when approximating (34) using $N = 1 0 0$ Monte Carlo samples. In particular, the picture reports the estimate mean values, together with their 99% confidence interval. Notably, the estimates obtained with POD-CNF perfectly agree with the ones obtained using the FOM: the two approaches, in fact, systematically yield overlapping confidence intervals. Moreover, the width of the latter is often comparable between the two approaches, further confirming the ability of POD-CNF in replicating variability of the FOM. Still, there are also major diferences amongst the diferent QoIs. For instance, POD-CNF clearly overestimates the uncertainties in $Q _ { 3 }$ and $Q _ { 4 }$ . We speculate that this is related to the very definition of the QoIs: some of them, in fact, correspond to nonlinear functionals that are not continuous in the $L ^ { 2 } \mathrm { - n o r m }$ . This makes (36) formally invalid (at least in the limi $N _ { h } \to + \infty )$ , thus hindering the advantages of probabilistic surrogates in general.

In principle, because the proposed surrogate is extremely eficient at inference time, we could consider improving our estimate by increasing the number of Monte Carlo samples in (35). However, we notice that this is only suitable up to a point, which is when the statistical error—related with the amplitude of the confidence interval—becomes comparable with the magnitude of the model error in (36). Circumventing this problem requires the development of suitable multi-fidelity strategies, specifically designed to balance these two sources of inaccuracy: see, e.g., [45, 16]. These methods, however, have been mostly developed for deterministic surrogates: extending the idea to probabilistic approaches is in order, but beyond the scope of this work.

Nonetheless, it can be insightful to estimate the potential reduction in computational cost. Table 6 shows the hypothetical running costs required to approximate (34) with a relative uncertainty, at 99% confidence, below 1%. The latter have been estimated by leveraging the previous pilot run and inverting the confidence interval formula; this made it possible to estimate the required sample size $N ^ { * }$ and the corresponding running costs. According to such projections, POD-CNF has the potential to reduce computational times by three orders of magnitude. This motivates developing the idea further, extending beyond this preliminary exploration.

## 5.2 Inverse UQ

In many applications, one is not only interested at predicting the response of a physical system for a prescribed problem configuration (forward problem), but rather aims at inferring the values of the problem parameters given, incomplete, experimental observations of the system itself (inverse problem). Because inverse problems are typically ill-posed, a common strategy is to tackle them using probabilistic methods based on Bayesian inference [22, 51]. In essence, rather than providing a pointwise estimate, these approaches leverage system measurements to produce a full probability distribution over the parameter space (the posterior distribution), thus quantifying uncertainties and providing additional insights.

In the case of intrinsic, structured stochasticities produced by hidden parameters, this task becomes particularly challenging. In fact, traditional approaches generally require modeling aleatoric uncertainty explicitly. This is perfectly doable when such uncertainty is purely associated with measurements noise, which can be reasonably described using additive models based on Gaussian assumptions. However, when uncertainties arise from within the system, the situation changes substantially, adding an additional layer of complexity.

To illustrate the idea, we shall present a simple example based on the linear elasticity problem discussed in Section 4.3. Recall that, there, $\mathbf { u } _ { \mu }$ represented the vector field associated with the local displacement of a solid body subject to an external load. The magnitude and the spatial extension of such load was determined by two parameters, m and $\delta ,$ respectively, collected as $\mu = [ m , \delta ]$ . The intrinsic uncertainty was generated by a hidden variable, $\theta \sim \mathcal { U } ( 0 , 2 \pi )$ , controlling the interface angle between two material layers constituting the body.

<table><tr><td>Label QoI</td><td></td><td>Expression</td><td>Continuous counterpart</td></tr><tr><td> $Q _ { 1 }$ </td><td>Average Concentration Gradient</td><td> $\begin{array} { r l } { } & { { } \frac { 1 } { N _ { c } } \sum _ { j = 1 } ^ { N _ { c } } \| \nabla \mathbf { u } _ { \mu } ^ { ( j ) } \| _ { 2 } } \end{array}$ </td><td> $\begin{array} { r } { \frac { 1 } { \left| \Omega \right| } \int _ { \Omega } \left| \left| \nabla u _ { \mu } \right| \right| d \pmb { x } } \end{array}$ </td></tr><tr><td> $Q _ { 2 }$ </td><td>Spatial Standard Deviation</td><td> $\begin{array} { r } { \left[ \frac { 1 } { N _ { h } } \sum _ { j = 1 } ^ { N _ { h } } | \mathrm { u } _ { \mu } ^ { ( j ) } - \overline { { u } } _ { \mu } | ^ { 2 } \right] ^ { 1 / 2 } } \end{array}$ </td><td> $\begin{array} { r } { \left[ \frac { 1 } { | \Omega | } \int _ { \Omega } | u _ { \mu } - \overline { { u } } _ { \mu } | ^ { 2 } d \pmb { x } \right] ^ { 1 / 2 } } \end{array}$ </td></tr><tr><td> $Q _ { 3 }$ </td><td>Relative Boundary Concentration</td><td> $\begin{array} { r l } { } & { { } \left[ \sum _ { j \in I _ { c _ { 3 } } } | \mathrm { u } _ { \pmb { \mu } } ^ { ( j ) } | \right] / \left[ \sum _ { i = 1 } ^ { 3 } \sum _ { j \in I _ { c _ { i } } } | \mathrm { u } _ { \pmb { \mu } } ^ { ( j ) } | \right] } \end{array}$ </td><td> $\begin{array} { r } { \left[ \int _ { \Gamma _ { c _ { 3 } } } \left| u _ { \pmb { \mu } } \right| d s \right] / \left[ \sum _ { i = 1 } ^ { 3 } \int _ { \Gamma _ { c _ { i } } } \left| u _ { \pmb { \mu } } \right| d s \right] } \end{array}$ </td></tr><tr><td> $Q _ { 4 }$ </td><td>High-Concentration Area Fraction</td><td> $\begin{array} { r } { \frac { 1 } { N _ { h } } \sum _ { j = 1 } ^ { N _ { h } } \mathbf { 1 } _ { ( 0 . 8 , + \infty ) } \big ( \mathrm { u } _ { \pmb { \mu } } ^ { ( j ) } \big ) } \end{array}$ </td><td> $\begin{array} { r } { \frac { 1 } { \left| \Omega \right| } \int _ { \Omega } \mathbf { 1 } _ { ( 0 . 8 , + \infty ) } ( u _ { \pmb { \mu } } ) d \pmb { x } } \end{array}$ </td></tr><tr><td> $Q _ { 5 }$ </td><td>Localized High-Concentration Event</td><td> $\begin{array} { r } { \mathbf { 1 } _ { ( 0 . 8 , + \infty ) } \big ( \operatorname* { m a x } _ { j \in I _ { \mathrm { r i g h t } } } \mathbf { u } _ { \pmb { \mu } } ^ { ( j ) } \big ) } \end{array}$ </td><td> ${ \mathbf { 1 } } _ { ( 0 . 8 , + \infty ) } \left( \operatorname* { s u p } _ { x \in \Omega _ { \mathrm { r i g h t } } } u _ { \pmb { \mu } } ( \pmb { x } ) \right)$ </td></tr></table>

Table 5. QoIs for the Forward UQ analysis. $| \Omega | = \mathrm { a r e a }$ of the spatial domain. $N _ { h } = \mathrm { n u m b e r }$ of vertices in the mesh. $N _ { c } =$ number of cells in the mesh. $\mathrm { u } ^ { ( j ) } = j \mathrm { t h }$ component of $\mathbf { u } _ { \mu } . \quad \nabla \mathbf { u } ^ { ( j ) }$ = value of the gradient of the FEM interpolant at the barycenter of the jth cell. $\overline { { u } } _ { \mu } =$ the nodal average (resp. integral) average of the concentration field (third and fourth columns, respectively). $\Gamma _ { c _ { i } }$ and $I _ { c _ { i } } = $ continuous boundary and discrete degrees of freedom associated with the i-th cylinder obstacle, respectively. $\Omega _ { \mathrm { r i g h t } } = \{ { \pmb x } \in \Omega : x _ { 1 } > 2 . 2 5 \}$ $I _ { \mathrm { r i g h t } } = \mathrm { d e g r e e s }$ of freedom in the mesh belonging to $\Omega _ { \mathrm { r i g h t } }$

![](images/9224f6e8fe9301a5871140ed54c5c60e2e871261fed0b13a526fa35608b1eedc.jpg)  
Figure 15. Comparison of the 99% confidence intervals obtained using the true FOM solver and the proposed POD-CNF surrogate for the evaluated Quantities of Interest (QoIs).

POD-CNF
<table><tr><td>QoI</td><td>Est. Mean</td><td>99% CI</td><td> $N ^ { * }$ </td><td> ${ \bf T i m e } ^ { * } { \bf \Xi } ( { \bf s } )$ </td></tr><tr><td> $Q _ { 1 }$ </td><td>0.6360</td><td>0.0138</td><td>472</td><td>0.8349</td></tr><tr><td> $Q _ { 2 }$ </td><td>0.2061</td><td>0.0040</td><td>376</td><td>0.6651</td></tr><tr><td> $Q _ { 3 }$ </td><td>0.3275</td><td>0.0073</td><td>501</td><td>0.8862</td></tr><tr><td> $Q _ { 4 }$ </td><td>0.1553</td><td>0.0248</td><td>25500</td><td>45.1065</td></tr><tr><td> $Q _ { 5 }$ </td><td>0.5700</td><td>0.1310</td><td>52564</td><td>92.9796</td></tr></table>

FOM
<table><tr><td>QoI</td><td>Est. Mean</td><td>99% CI</td><td> $N ^ { * }$ </td><td> ${ \bf T i m e } ^ { * } { \bf \Xi } ( { \bf s } )$ </td></tr><tr><td> $Q _ { 1 }$ </td><td>0.6486</td><td>0.0140</td><td>464</td><td>478.5703</td></tr><tr><td> $Q _ { 2 }$ </td><td>0.2083</td><td>0.0030</td><td>213</td><td>219.6885</td></tr><tr><td> $Q _ { 3 }$ </td><td>0.3302</td><td>0.0020</td><td>36</td><td>37.1305</td></tr><tr><td> $Q _ { 4 }$ </td><td>0.1495</td><td>0.0132</td><td>7771</td><td>8015.0204</td></tr><tr><td> $Q _ { 5 }$ </td><td>0.4400</td><td>0.0248</td><td>88680</td><td>91464.6777</td></tr></table>

Table 6. Summary of the Forward UQ analysis for POD-CNF (left) and FOM (right). Est. Mean = value of the Monte Carlo estimator with $N = 1 0 0$ samples. 99% CI = semi-amplitude of the 99% confidence interval with N = 100 Monte Carlo samples. $N ^ { * } =$ estimated sample size required to reduce the relative uncertainty below 1%. Time<sup>∗</sup> = running time (in seconds) required by the model to generate N $N ^ { * }$ samples.

We consider the problem of inferring the values of m and δ given sensor measurements of the displacement vector field at 31 equally spaced sensors located at the top surface of the body. We shall write $\mathbf { u } _ { \mathrm { s e n s } } \in \mathbb { R } ^ { 6 2 }$ to the denote the data vector collecting such measurements. Assuming that all stochasticites arise from the hidden variable $\theta ,$ , whose value is unknown, we shall neglect potential measurement noise afecting the sensors. This is depart from the conventional setting and focus entirely on the new challenges posed by the considered framework.

Following a Bayesian paradigm, we equip the parameter space with a prior probability measure $\mathsf { p } ( \mu )$ , chosen to be the uniform distribution over $\mathcal { P }$ . According to Bayes’ rule, the posterior distribution $\mathsf { p } ( \mu \mid \mathbf { u } _ { \mathrm { s e n s } } )$ satisfies

$$
\mathsf { p } ( \mu \mid \mathbf { u } _ { \mathrm { s e n s } } ) \propto \mathsf { p } ( \mathbf { u } _ { \mathrm { s e n s } } \mid \mu ) \mathsf { p } ( \mu ) .\tag{37}
$$

In standard settings, the likelihood function $\mathsf { p } ( \mathbf { u } _ { \mathrm { s e n s } } \mid \mu )$ can be readily derived from the model used to described measurement uncertainties. In this case, however, we cannot express the likelihood function explicitly, as the latter depends intrinstically on the PDE solver and its interaction with the hidden variable $\theta \sim \mathcal { U } ( 0 , 2 \pi )$

However, we notice that, because our trained POD-CNF is able to approximate the conditional law of ${ \bf u } \mid \mu .$ , we can leverage this knowledge to surrogate $\mathsf { p } ( \mathbf { u } _ { \mathrm { s e n s } } \mid \mu )$ . To see this, let $\mathbf { P } _ { \mathrm { s e n s } } \in \mathbf { \bar { \mathbb { R } } } ^ { 6 2 \times N _ { h } }$ the matrix mapping full displacement vectors onto sensor measurements. Then

$$
{ \bf u } _ { \mathrm { s e n s } } \mid \mu = ( { \bf P } _ { \mathrm { s e n s } } \mathbf { u } ) \mid \mu \stackrel { d } { \approx } { \bf P } _ { \mathrm { s e n s } } { \bf V } \phi ( \mu , z ) = : { \bf V } _ { \mathrm { s e n s } } \phi ( \mu , z ) ,\tag{38}
$$

where $\overset { d } { \approx }$ denotes approximation in distribution and $\mathbf { V } _ { \mathrm { s e n s } } : = \mathbf { P } _ { \mathrm { s e n s } } \mathbf { V }$ . While (38) does not sufice to approximate $\mathsf { p } ( \mathbf { u } _ { \mathrm { s e n s } } \mid \mu )$ explicitly, it can be used to estimate the distribution numerically. In fact, because $\phi$ is computationally inexpensive to evaluate, the law of $\mathbf { V } _ { \mathrm { s e n s } } \phi ( \pmb { \mu } , z )$ , and thus that of $\mathbf { u } _ { \mathrm { s e n s } } \mid \mu ,$ , can be estimated empirically by drawing $N _ { g e n }$ i.i.d. samples and fitting a Kernel Density Estimation (KDE) model. This procedure can be then integrated within a suitable Markov chain Monte Carlo (MCMC) method to approximate the posterior distribution using (37).

Figure 16 shows the results obtained using this strategy in two diferent parametric scenarios. The pipeline has been implemented using an adaptive Metropolis-Hastings algorithm [25], complemented by a two-phase strategy based on an initial posterior exploration (chain with higher temperature and fixed proposal) followed by a subsequent refinement phase (lower temperature and adaptive proposal). During the iterations of the MCMC algorithm, the likelihood function has been estimated using the aforementioned KDE approximation, here implemented using a Gaussian kernel. For the latter, we used $N _ { g e n } = 1 0 0$ and a bandwidth of $h = 0 . 1$ during the exploratory phase of the MCMC, and $N _ { g e n } = 2 0 0$ with $h = 0 . 0 3$ during the refinement stage.

As show in Figure 16 and Table 7 the proposed strategy successfully estimates the loading parameters m and $\delta ,$ returning a posterior concentrated near the ground truth values. The approximation is more accurate for m (pointwise error below 2%) and slightly less for δ (pointwise errors around 3%-7%, but with true values always falling in the proposed credibility intervals).

To corroborate the analysis, we repeated the pipeline using the FOM solver, but introducing suitable modifications that would keep the overall computational cost comparable to that of POD-CNF. Specifically, we reduced the number of iterations in the MCMC by an order of magnitude (1000 iterations for the exploratory phase and 3000 for the refinement stage), as well as the number of samples generated per iteration $( N _ { g e n } = 1 0 , 2 5$ in the two phases). Even with these modifications, the FOM analysis remained substantially more intensive than the previous one (approximately one hour versus 1–6 minutes), but was at least feasible to run.

As shown in Figure 16 the results are consistent with the previous analysis. In the first scenario, the two approaches yield posterior distributions localized around the same maximum value, although their spread difers substantially. Such diference could be caused by an intrinsic discrepancy between the two or by the diferent hyperparameters used for the MCMC algorithm. In the second scenario, instead, the two distributions identify a similar high probability region, presenting an elongated shape that extends towards the top-right corner of the parameter space.

Tables 7-8 provide a broader overview, extending the comparison to other parametric scenarios. When both models demonstrate strong predictive performance, their 95% credibility intervals typically exhibit a significant overlap. In particular, the intervals generated by POD-CNF surrogate tend to intersect with the tighter ones produced by the FOM, indicating an agreement between the high-probability regions predicted by the two approaches. At the same time, the pointwise estimate produced by the FOM are not always reliable, and the tightness of such intervals turns out to be unjustified: see, $\mathrm { e . g . }$ , rows $5 ,$ 6 and 8 in Table 8. In these challenging cases, the FOM yields relative errors on m and δ exceeding 40% and 30%, respectively. In contrast, the adaptive Metropolis-Hastings algorithm driven by the POD-CNF surrogate remains stable, successfully recovering true parameter values in most tested configurations at a fraction of the computational time.

![](images/0b3239cf3cc80c2d2b2ef8261396d69592209aca33a227393e4692ba2483ff0c.jpg)  
Figure 16. Linear elasticity problem, §4.3. For both panels, the first row reports the MCMC traces and posterior distribution $\mathsf { p } ( \mu \mid \mathrm { \mathbf { u } } _ { \mathrm { s e n s } } )$ from the high-fidelity numerical solver (FOM baseline), while the second row shows the corresponding results obtained via the POD-CNF surrogate.

<table><tr><td>True m</td><td>Est. m</td><td>95% CI</td><td>Error</td><td>True δ</td><td>Est. δ</td><td>95% CI</td><td>Error</td><td>Time</td></tr><tr><td>1.7384</td><td>1.7412</td><td>[1.5438, 1.9414]</td><td>0.16%</td><td>0.2295</td><td>0.2127</td><td>[0.1869, 0.2385]</td><td>7.34%</td><td>42s</td></tr><tr><td>1.6228</td><td>1.5195</td><td>[1.4951, 1.7095]</td><td>6.37%</td><td>0.1653</td><td>0.2088</td><td>[0.1449, 0.2270]</td><td>26.35%</td><td>46s</td></tr><tr><td>1.1777</td><td>1.3440</td><td>[1.0607, 1.4985]</td><td>14.12%</td><td>0.2032</td><td>0.1665</td><td>[0.0904, 0.2408]</td><td>18.08%</td><td>45s</td></tr><tr><td>1.8151</td><td>1.8463</td><td>[1.7462, 1.9728]</td><td>1.72%</td><td>0.2354</td><td>0.2380</td><td>[0.2220, 0.2493]</td><td>1.12%</td><td>6m 20s</td></tr><tr><td>1.2920</td><td>1.3550</td><td>[1.1414, 1.5936]</td><td>4.88%</td><td>0.2278</td><td>0.2109</td><td>[0.1703,0.2441]</td><td>7.42%</td><td>6m 15s</td></tr><tr><td>1.0757</td><td>1.1165</td><td>[1.0077, 1.2689]</td><td>3.79%</td><td>0.1513</td><td>0.1463</td><td>[0.1198,0.1734]</td><td>3.27%</td><td>42s</td></tr><tr><td>1.1656</td><td>1.1863</td><td>[1.0111,1.4722]</td><td>1.78%</td><td>0.1905</td><td>0.1838</td><td>[0.1509,0.2177]</td><td>3.52%</td><td>41s</td></tr><tr><td>1.2045</td><td>1.2320</td><td>[1.0326, 1.4742]</td><td>2.29%</td><td>0.2188</td><td>0.2132</td><td>[0.1740, 0.2453]</td><td>2.53%</td><td>6m 3s</td></tr></table>

Table 7. Summary of the inverse problem results for the POD-CNF surrogate model. CI = marginal credibility interval.

<table><tr><td>True m</td><td>Est. m</td><td>95% CI</td><td>Error</td><td>True δ</td><td>Est. δ</td><td>95% CI</td><td>Error</td><td>Time</td></tr><tr><td>1.7384</td><td>1.7383</td><td>[1.6788, 1.8011]</td><td>0.01%</td><td>0.2295</td><td>0.2178</td><td>[0.2012,0.2325]</td><td>5.08%</td><td>63m 20s</td></tr><tr><td>1.6228</td><td>1.4549</td><td>[1.3883, 1.5074]</td><td>10.35%</td><td>0.1653</td><td>0.1835</td><td>[0.1676,0.1994]</td><td>11.01%</td><td>61m 40s</td></tr><tr><td>1.1777</td><td>1.3873</td><td>[1.3118, 1.4656]</td><td>17.80%</td><td>0.2032</td><td>0.1829</td><td>[0.1673,0.1989]</td><td>10.01%</td><td>65m 53s</td></tr><tr><td>1.8151</td><td>1.8221</td><td>[1.7673, 1.8757]</td><td>0.39%</td><td>0.2354</td><td>0.2414</td><td>[0.2338,0.2496]</td><td>2.57%</td><td>65m 0s</td></tr><tr><td>1.2920</td><td>1.8346</td><td>[1.7432, 1.9279]</td><td>41.99%</td><td>0.2278</td><td>0.1497</td><td>[0.1342,0.1656]</td><td>34.28%</td><td>60m 50s</td></tr><tr><td>1.0757</td><td>1.9002</td><td>[1.7527, 1.9978]</td><td>76.64%</td><td>0.1513</td><td>0.0846</td><td>[0.0678, 0.0993]</td><td>44.06%</td><td>63m 54s</td></tr><tr><td>1.1656</td><td>1.0948</td><td>[1.0016, 1.2011]</td><td>6.07%</td><td>0.1905</td><td>0.1952</td><td>[0.1682,0.2304]</td><td>2.48%</td><td>57m 12s</td></tr><tr><td>1.2045</td><td>1.7189</td><td>[1.6497, 1.8027]</td><td>42.71%</td><td>0.2188</td><td>0.1501</td><td>[0.1345,0.1654]</td><td>31.40%</td><td>62m 30s</td></tr></table>

Table 8. Summary of the inverse problem results for the FOM baseline. CI = marginal credibility interval.

The proposed analysis, however, is not free from limitations and only constitutes a preliminary exploration. Indeed, the necessity to approximate the likelihood function empirically and the subsequent the reliance on KDE currently pose major limitations, hindering the scalability of the approach to more complex inverse problems. Similarly, the comparison between the proposed approach and the FOM reference is not entirely exhaustive. These aspects could be addressed by consider more refined inference strategies that combine ideas from Bayesian statistics and Simulation Based Inference [14]. Still, because Inverse UQ is an application of interest but not the main focus of this work, we leave these considerations for future explorations.

## 6 Conclusions

In this paper, we presented POD-CNF, a hybrid generative framework designed to tackle the challenges associated with hidden parameters in high-dimensional systems governed by stochastic PDEs, and the corresponding Uncertainty Quantification (UQ) tasks. By integrating linear dimensionality reduction via Proper Orthogonal Decomposition (POD) with the expressive density estimation capabilities of Conditional Normalizing Flows, we developed a too capable of modeling the full conditional probability distribution of the physical field, p(u|µ), thus accounting for the aleatoric uncertainty induced by the unobserved latent variables.

The numerical experiments demonstrated the efectiveness of the proposed approach, further consolidating our theoretical analysis. A key indicator of success was the Wasserstein-1 distance (W ) between the generated and ground-truth distributions. In all test cases, the global W error was significantly lower than the one obtained by the benchmark models, including POD-GP and deterministic baselines, confirming the ability of POD-CNF to capture complex, non-Gaussian marginals rather than merely regressing a deterministic average. The data eficiency of the proposed strategy was assessed through a dedicated sensitivity analysis, showing robust performances even when reducing the number of training samples by a significant amount (-75% with respect to the original sample size).

A major advantage of the POD-CNF framework lies in its generative capabilities. These allowed us to seamlessly extend the method to tackle not only forward UQ tasks, but also inverse UQ problems related with parameter estimation. By embedding the pre-trained generative model within a Bayesian inference framework, we were able to approximate the posterior distribution of the physical parameters µ through a specialized Markov chain Monte Carlo sampling.

Despite these promising results, the current methodology also presents some limitations:

• relying on POD implies a linear approximation of the solution manifold, which may struggle with, e.g., advection dominated phenomena or moving discontinuities, especially if considering time-dependent problems. Future developments could replace POD with nonlinear dimensionality reduction techniques, such as variational autoencoders (VAEs) [49], which can compress these complex manifolds more efectively;

• the proposed method for constructing probabilistic surrogates, and tackling UQ tasks, is purely data-driven; it learns the conditional distribution of the POD coeficients, yielding physically consistent outputs (as demonstrated by our numerical experiments), but without enforcing the underlying governing equations. Future work could incorporate physics-informed regularization or even hard constraints [42]: for instance, by adding to the loss function a penalty term involving the PDE residual $\| \mathcal { R } ( \hat { \mathbf { u } } ) \| ^ { 2 }$ , the model could be encouraged to generate physically consistent solutions without strictly requiring a solver during inference;

• our preliminary exploration concerning forward and inverse UQ tasks remains subject to limitations. In the first case, when estimating conditional expectations of given quantities of interest, increasing the number of Monte Carlo samples reduces the sampling error but it does not address surrogate model error, motivating the development of suitable multi-fidelity strategies for probabilistic surrogates. In the inverse UQ analysis, instead, the major limitation is given by the empirical approximation of the likelihood function via Kernel Density Estimation, which currently limits the scalability of the approach to more complex problems.

On top of this, a significant avenue for future research involves extending our framework to time-dependent problems. This would require modeling the temporal evolution of the latent coeficients c(t) by learning either their transition probabilities or the distribution of their trajectories, ultimately enabling the prediction of time-dependent quantities of interest under uncertainty.

## Acknowledgments

AV, NRF and AM acknowledge the Project "Reduced Order Modeling and Deep Learning for the real-time approximation of PDEs (DREAM)" (Starting Grant No. FIS00003154), funded by the Italian Science Fund (FIS) - Ministero dell’Università e della Ricerca (MUR). NRF and AM are members of GNCS (Gruppo Nazionale per il Calcolo Scien tifico) and gratefully acknowledge its support. The present research is also part of the activities of project Dipartimento di Eccellenza 2023-2027, Department of Mathematics, Politecnico di Milano, funded by MUR.

## References

[1] Albergo, M., Bofi, N.M., Vanden-Eijnden, E., 2025. Stochastic interpolants: A unifying framework for flows and difusions. Journal of Machine Learning Research 26, 1–80.

[2] Berman, J., Blickhan, T., Peherstorfer, B., 2026. Stochastic lifting for generating trajectories of stochastic physical systems, in: Proceedings of the 43rd International Conference on Machine Learning, PMLR. pp. 7657–7682.

[3] Billaud-Friess, M., Macherey, A., Nouy, A., Prieur, C., 2024. A probabilistic reduced basis method for parameter dependent problems. Advances in Computational Mathematics 50, 19.

[4] Boon, W.M., Franco, N.R., Fumagalli, A., Zunino, P., 2025. Deep learning-based reduced-order modeling of darcy-flow systems with local mass conservation. International Journal for Numerical Methods in Engineering 126, e70188.

[5] Boyaval, S., Le Bris, C., Lelievre, T., Maday, Y., Nguyen, N.C., Patera, A.T., 2010. Reduced basis techniques for stochastic problems. Archives of Computational methods in Engineering 17, 435–454.

[6] Brivio, S., Fresca, S., Franco, N.R., Manzoni, A., 2024. Error estimates for pod-dl-roms: a deep learning framework for reduced order modeling of nonlinear parametrized pdes enhanced by proper orthogonal decomposition. Advances in Computational Mathematics 50, 33.

[7] Bülte, C., Scholl, P., Kutyniok, G., 2025. Probabilistic neural operators for functional uncertainty quantification. arXiv preprint arXiv:2502.12902 .

[8] Chen, P., Quarteroni, A., Rozza, G., 2013. A weighted reduced basis method for elliptic partial diferential equations with random input data. SIAM Journal on Numerical Analysis 51, 3163–3185.

[9] Chen, P., Quarteroni, A., Rozza, G., 2014. Comparison between reduced basis and stochastic collocation methods for elliptic problems. Journal of Scientific Computing 59, 187–216.

[10] Chen, P., Quarteroni, A., Rozza, G., 2017. Reduced basis methods for uncertainty quantification. SIAM/ASA Journal on Uncertainty Quantification 5, 813–869.

[11] Cockayne, J., Oates, C.J., Sullivan, T.J., Girolami, M., 2019. Bayesian probabilistic numerical methods. SIAM Review 61, 756–789.

[12] Cohen, A., 2021. From linear to nonlinear n-width: optimality in reduced modelling. Journées “Réduction de modèles et traitement géométrique des données”, IHP .

[13] Coscia, D., Demo, N., Rozza, G., 2024. Generative adversarial reduced order modelling. Scientific Reports 14, 3826.

[14] Cranmer, K., Brehmer, J., Louppe, G., 2020. The frontier of simulation-based inference. Proceedings of the National Academy of Sciences 117, 30055–30062.

[15] Dinh, L., Sohl-Dickstein, J., Bengio, S., 2016. Density estimation using real nvp. arXiv preprint arXiv:1605.08803

[16] Farcas, I.G., Peherstorfer, B., Neckel, T., Jenko, F., Bungartz, H.J., 2023. Context-aware learning of hierarchies of low-fidelity models for multi-fidelity uncertainty quantification. Computer Methods in Applied Mechanics and Engineering 406, 115908.

[17] Fjordholm, U.S., Lanthaler, S., Mishra, S., 2017. Statistical solutions of hyperbolic conservation laws: Foundations. Archive for Rational Mechanics and Analysis 226, 809–849.

[18] Foias, C., 1972. Statistical study of navier-stokes equations, i. Rendiconti del Seminario Matematico della Uni versita di Padova 48, 219–348.

[19] Franco, N., Manzoni, A., Zunino, P., 2023. A deep learning approach to reduced order modelling of parameter dependent partial diferential equations. Mathematics of Computation 92, 483–524.

[20] Franco, N.R., Manzoni, A., Zunino, P., Hesthaven, J.S., 2026. Deep orthogonal decomposition: a continuously adaptive neural network approach to model order reduction of parametrized partial diferential equations. Advances in Computational Mathematics 52, 31.

[21] Fresca, S., Manzoni, A., 2022. Pod-dl-rom: Enhancing deep learning-based reduced order models for nonlinear parametrized pdes by proper orthogonal decomposition. Computer Methods in Applied Mechanics and Engineering 388, 114181.

[22] Gelman, A., Carlin, J.B., Stern, H.S., Rubin, D.B., 1995. Bayesian data analysis. Chapman and Hall/CRC.

[23] Goodfellow, I., Bengio, Y., Courville, A., 2016. Deep Learning. MIT Press.

[24] Guo, M., Hesthaven, J.S., 2018. Reduced order modeling for nonlinear structural analysis using gaussian process regression. Computer methods in applied mechanics and engineering 341, 807–826.

[25] Haario, H., Saksman, E., Tamminen, J., 2001. An adaptive metropolis algorithm .

[26] Hennig, P., Osborne, M.A., Kersting, H.P., 2022. Probabilistic Numerics. Cambridge University Press.

[27] Hesthaven, J.S., Pagliantini, C., Rozza, G., 2022. Reduced basis methods for time-dependent problems. Acta Numerica 31, 265–345.

[28] Hesthaven, J.S., Rozza, G., Stamm, B., 2015. Certified Reduced Basis Methods for Parametrized Partial Difer ential Equations. SpringerBriefs in Mathematics, Springer International Publishing, Cham, Switzerland.

[29] Hesthaven, J.S., Ubbiali, S., 2018. Non-intrusive reduced order modeling of nonlinear problems using neural networks. Journal of Computational Physics 363, 55–78.

[30] Kim, H., Zhang, L., Friend, J., Kramer, B., 2026. Stochastic operator inference for reduced-order modeling of capillary wave turbulence using experimental measurements. arXiv preprint arXiv:2609.04008 .

[31] Kingma, D.P., Ba, J., 2015. Adam: A method for stochastic optimization, in: International Conference on Learning Representations, San Diego, CA, USA.

[32] Koehler, F., Mehta, V., Risteski, A., 2021. Representational aspects of depth and conditioning in normalizing flows, in: International Conference on Machine Learning, PMLR. pp. 5628–5636.

[33] Kotowski, D., Ohlberger, M., 2026. A new adaptive deep learning based reduced order model for hybrid-type parabolic pdes: Rigorous error analysis and applications. arXiv preprint arXiv:2604.22512 .

[34] Law, K., Stuart, A., Zygalakis, K., Stuart, A., 2015. Data assimilation: a mathematical introduction. volume 62. Springer.

[35] Lee, H., Pabbaraju, C., Sevekari, A.P., Risteski, A., 2021. Universal approximation using well-conditioned normalizing flows. Advances in Neural Information Processing Systems 34, 12700–12711.

[36] Lim, J.H., Kovachki, N.B., Baptista, R., Beckham, C., Azizzadenesheli, K., Kossaifi, J., Voleti, V., Song, J., Kreis, K., Kautz, J., et al., 2025. Score-based difusion models in function space. Journal of Machine Learning Research 26, 1–62.

[37] Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M., 2023. Flow matching for generative modeling, in: The Eleventh International Conference on Learning Representations. URL: https://openreview.net/forum?id=PqvMRDCJT9t.

[38] Lu, L., Meng, X., Cai, S., Mao, Z., Goswami, S., Zhang, Z., Karniadakis, G.E., 2022. A comprehensive and fair comparison of two neural operators (with practical extensions) based on fair data. Computer Methods in Applied Mechanics and Engineering 393, 114778.

[39] Lyu, J., Chen, Z., Feng, C., Cun, W., Zhu, S., Geng, Y., Xu, Z., Yongwei, C., 2022. Para-cflows: c<sup>k</sup>-universal difeomorphism approximators as superior neural surrogates. Advances in Neural Information Processing Systems 35, 28829–28841.

[40] Maday, Y., Rønquist, E.M., 2002. A reduced-basis element method. Journal of scientific computing 17, 447–459.

[41] Molinaro, R., Lanthaler, S., Raonić, B., Rohner, T., Armegioiu, V., Simonis, S., Grund, D., Ramic, Y., Wan, Z.Y., Sha, F., et al., 2024. Generative ai for fast and accurate statistical computation of fluids. arXiv preprint arXiv:2409.18359 .

[42] Mücke, N.T., Sanderse, B., 2026. Physics-aware generative models for turbulent fluid flows through energyconsistent stochastic interpolants. Computers & Fluids , 107098.

[43] Nair, P.B., Keane, A.J., 2002. Stochastic reduced basis methods. AIAA journal 40, 1653–1664.

[44] Papamakarios, G., Nalisnick, E., Rezende, D.J., Mohamed, S., Lakshminarayanan, B., 2021. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research 22, 1–64.

[45] Peherstorfer, B., Willcox, K., Gunzburger, M., 2016. Optimal model management for multifidelity monte carlo estimation. SIAM Journal on Scientific Computing 38, A3163–A3194.

[46] Pichi, F., Ballarin, F., Rozza, G., Hesthaven, J.S., 2023. An artificial neural network approach to bifurcating phenomena in computational fluid dynamics. Computers & Fluids 254, 105813.

[47] Quarteroni, A., 2009. Numerical models for diferential problems. Springer.

[48] Quarteroni, A., Manzoni, A., Negri, F., 2015. Reduced basis methods for partial diferential equations: an introduction. Springer.

[49] Rezende, D., Mohamed, S., 2015. Variational inference with normalizing flows, in: International conference on machine learning, PMLR. pp. 1530–1538.

[50] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B., 2022. High-resolution image synthesis with latent difusion models, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695.

[51] Stuart, A.M., 2010. Inverse problems: a bayesian perspective. Acta numerica 19, 451–559.

[52] Villani, C., et al., 2009. Optimal transport: old and new. volume 338. Springer.

[53] Wang, Q., Hesthaven, J.S., Ray, D., 2019. Non-intrusive reduced order modeling of unsteady flows using artificial neural networks with application to a combustion problem. Journal of computational physics 384, 289–307.

[54] Williams, D., 1991. Probability with martingales. Cambridge university press.

## A Proofs of Section 3.1

Proof of Lemma 1. Fix any $\pmb { \mu } \in \mathcal { P }$ . Using the Kantorovich-Rubinstein duality [52, Chapter 5], we can write the Wasserstein-1 distance between the distributions of $\mathbf { u } _ { \mu }$ and $\mathbf { V } \phi ( \pmb { \mu } , z )$ as

$$
\mathcal { W } _ { 1 } ( \mathbf { u } _ { \mu } , \mathbf { V } \phi ( \pmb { \mu } , z ) ) = \operatorname* { s u p } _ { Q \in \mathcal { B } _ { 1 } ( N _ { h } ) } \left| \mathbb { E } _ { \mathbf { u } _ { \mu } } [ Q ( \mathbf { u } _ { \mu } ) ] - \mathbb { E } _ { z } [ Q ( \mathbf { V } \phi ( \pmb { \mu } , z ) ) ] \right|\tag{39}
$$

where $\mathcal { B } _ { 1 } ( m )$ denotes the set of all Lipschitz continuous maps $Q : \mathbb { R } ^ { m } $ R with Lipschitz constant no larger than 1. Notice that the two expectations in (39) are taken with respect to $\mathbf { u } _ { \mu }$ and $z ,$ respectively. We now notice that, for every $Q \in \mathcal { B } _ { 1 } ( N _ { h } )$ , by the triangle inequality

$$
\big | \mathbb { E } _ { \mathbf { u } _ { \mu } } [ Q ( \mathbf { u } _ { \mu } ) ] - \mathbb { E } _ { z } [ Q ( \mathbf { V } \phi ( \mu , z ) ) ] \big | \leq \big | \mathbb { E } _ { \mathbf { u } _ { \mu } } [ Q ( \mathbf { u } _ { \mu } ) ] - \mathbb { E } _ { \mathbf { u } _ { \mu } } [ Q ( \mathbf { V } \mathbf { V } ^ { \top } \mathbf { u } _ { \mu } ) ] \big | + \big | \mathbb { E } _ { \mathbf { u } _ { \mu } } [ Q ( \mathbf { V } \mathbf { V } ^ { \top } \mathbf { u } _ { \mu } ) ] - \mathbb { E } _ { z } [ Q ( \mathbf { V } \phi ( \mu , z ) ) ] \big | .\tag{40}
$$

The first term in the above can be bounded as

$$
\begin{array} { r } { \left| \mathbb { E } _ { { \mathbf { u } } _ { \mu } } [ Q ( { \mathbf { u } } _ { \mu } ) ] - \mathbb { E } _ { { \mathbf { u } } _ { \mu } } [ Q ( { \mathbf { V } } { \mathbf { V } } ^ { \top } { \mathbf { u } } _ { \mu } ) ] \right| \leq \mathbb { E } _ { { \mathbf { u } } _ { \mu } } \left[ \left| Q ( { \mathbf { u } } _ { \mu } ) - Q ( { \mathbf { V } } { \mathbf { V } } ^ { \top } { \mathbf { u } } _ { \mu } ) \right| \right] \leq \mathbb { E } _ { { \mathbf { u } } _ { \mu } } \left\| { \mathbf { u } } _ { \mu } - { \mathbf { V } } { \mathbf { V } } ^ { \top } { \mathbf { u } } _ { \mu } \right\| , } \end{array}\tag{41}
$$

since $Q \in \mathcal { B } _ { 1 } ( N _ { h } )$ . The second term in Eq. (40), instead, can be re-written as

$$
\bigl | \mathbb { E } _ { { \mathbf { u } } _ { \mu } } [ Q ( { \mathbf { V } } { \mathbf { V } } ^ { \top } { \mathbf { u } } _ { \mu } ) ] - \mathbb { E } _ { z } [ Q ( { \mathbf { V } } \phi ( \mu , z ) ) ] \bigr | = \bigl | \mathbb { E } _ { { \mathbf { u } } _ { \mu } } [ q ( { \mathbf { V } } ^ { \top } { \mathbf { u } } _ { \mu } ) ] - \mathbb { E } _ { z } [ q ( \phi ( \mu , z ) ) ] \bigr | ,
$$

where $q : \mathbb { R } ^ { n }  \mathbb { R }$ is given as $q ( \mathbf { c } ) : = Q ( \mathbf { V } \mathbf { c } )$ . It is straightforward to see that $q \in \mathcal { B } _ { 1 } ( n )$ . In fact, because V is orthonormal, for every $\mathbf { c } , \mathbf { c } ^ { \prime } \in \mathbb { R } ^ { n }$ one has $| q ( \mathbf { c } ) - q ( \mathbf { c } ^ { \prime } ) | = | Q ( \mathbf { V } \mathbf { c } ) - Q ( \mathbf { V } \mathbf { c } ^ { \prime } ) | \leq \| \mathbf { V } \mathbf { c } - \mathbf { V } \mathbf { c } ^ { \prime } \| = | \mathbf { V } ( \mathbf { c } - \mathbf { c } ^ { \prime } ) | | = \| \mathbf { c } - \mathbf { c } ^ { \prime } \|$ It follows that

$$
\bigl | \mathbb { E } _ { \mathbf { u } _ { \mu } } \bigl [ q ( \mathbf { V } ^ { \top } \mathbf { u } _ { \mu } ) \bigr ] - \mathbb { E } _ { z } \bigl [ q ( \phi ( \mu , z ) ) \bigr ] \bigr | \leq \operatorname* { s u p } _ { \tilde { q } \in \mathcal { B } _ { 1 } ( n ) } \bigl | \mathbb { E } _ { \mathbf { u } _ { \mu } } \bigl [ \tilde { q } ( \mathbf { V } ^ { \top } \mathbf { u } _ { \mu } ) \bigr ] - \mathbb { E } _ { z } \bigl [ \tilde { q } ( \phi ( \mu , z ) ) \bigr ] \bigr | = \mathcal { W } _ { 1 } ( \mathbf { V } ^ { \top } \mathbf { u } _ { \mu } , \phi ( \mu , z ) ) .\tag{42}
$$

Therefore, combining (40) with (41) and (42) we obtain

$$
\begin{array} { r } { \left| \mathbb { E } _ { { \mathbf { u } } _ { \mu } } [ Q ( { \mathbf { u } } _ { \mu } ) ] - \mathbb { E } _ { z } [ Q ( { \mathbf { V } } \phi ( { \pmb { \mu } } , z ) ) ] \right| \leq \mathbb { E } _ { { \mathbf { u } } _ { \mu } } \left\| { \mathbf { u } } _ { \mu } - { \mathbf { V } } { \mathbf { V } } ^ { \top } { \mathbf { u } } _ { \mu } \right\| + \mathcal { W } _ { 1 } \big ( { \mathbf { V } } ^ { \top } { \mathbf { u } } _ { \mu } , \phi ( \mu , z ) \big ) . } \end{array}
$$

Since $Q \in \mathcal { B } _ { 1 } ( N _ { h } )$ was arbitrary, passing at the supremum yields the desired conclusion.

Proof of Corollary 1. By integrating (15) with respect to $\textstyle \mu ,$ we can bound the left-hand-side in Eq. (16) with $\begin{array} { r } { \int _ { \mathcal { P } } \mathbb { E } \| \mathbf { u } _ { \mu } - \mathbf { V } \mathbf { V } ^ { \top } \mathbf { u } _ { \mu } \| \xi ( \mu ) d \mu + \int _ { \mathcal { P } } \mathcal { W } _ { 1 } ( \mathbf { V } ^ { \top } \mathbf { u } _ { \mu } , \mathbf { \xi } \phi ( \mu , z ) ) \xi ( \mu ) d \mu . } \end{array}$ By Jensen’s inequality, the first term can be bounded further as

$$
\int _ { \mathcal { P } } \mathbb { E } \| \mathbf { u } _ { \mu } - \mathbf { V } \mathbf { V } ^ { \top } \mathbf { u } _ { \mu } \| \xi ( \mu ) d \mu \leq \left[ \int _ { \mathcal { P } } \mathbb { E } \| \mathbf { u } _ { \mu } - \mathbf { V } \mathbf { V } ^ { \top } \mathbf { u } _ { \mu } \| ^ { 2 } \xi ( \mu ) d \mu \right] ^ { 1 / 2 } .
$$

However, by standard arguments based on the Eckart-Young-Schmidt Theorem, the POD matrix V satisfies $\int _ { \mathcal { P } } \mathbb { E } \| \mathbf { u } _ { \mu } -$ $\begin{array} { r } { \mathbf { V } \mathbf { V } ^ { \top } \mathbf { u } _ { \pmb { \mu } } \| ^ { 2 } \xi ( { \pmb { \mu } } ) d { \pmb { \mu } } = \sum _ { j > n } \lambda _ { j } } \end{array}$ (see, e.g. [48, Proposition 6.3]), thus yielding the claimed inequality. □

Proof of Theorem 1. We first notice that, since V is orthonormal and the random vectors $\mathbf { u } _ { \mu }$ are uniformly bounded, one has $\| \mathbf { V } ^ { \top } \mathbf { u } _ { \mu } \| \leq R$ almost surely for all $\pmb { \mu } \in \mathcal { P }$ . In particular, being $B ( 0 , R ) \subset \mathbb { R } ^ { n }$ the ball of radius R centered at the origin, one has $\cup _ { \pmb { \mu } \in \mathcal { P } ^ { \mathrm { S u p p } } } ( \mathbb { P } _ { n , \pmb { \mu } } ) \subseteq B ( 0 , R )$ , meaning that the union of the supports of the reduced

distributions is bounded. We then notice that the map $\mu \to \mathbb { P } _ { n , \mu }$ is continuous in the $\mathcal { W } _ { 1 }$ metric. Indeed, since $Q = q \circ \mathbf { V } ^ { \intercal }$ is 1-Lipschitz wheneve $q : \mathbb { R } ^ { n }  \mathbb { R }$ is 1-Lipschitz, one has

$$
\mathcal { W } _ { 1 } ( { \mathbb P } _ { n , \mu } , \mathbb P _ { n , \mu ^ { \prime } } ) = \operatorname* { s u p } _ { q \in \mathcal { B } _ { 1 } ( n ) } \left| \mathbb E [ q ( \nabla ^ { \top } \mathbf u _ { \mu } ) ] - \mathbb E [ q ( \nabla ^ { \top } \mathbf u _ { \mu ^ { \prime } } ) ] \right| \leq \operatorname* { s u p } _ { Q \in \mathcal { B } _ { 1 } ( N _ { h } ) } \left| \mathbb E [ Q ( \mathbf u _ { \mu } ) ] - \mathbb E [ Q ( \mathbf u _ { \mu ^ { \prime } } ) ] \right| = \mathcal { W } _ { 1 } ( \mathbb P _ { \mu } , \mathbb P _ { \mu ^ { \prime } } )
$$

where, as before, $\mathcal { B } _ { 1 } ( m )$ denotes the set of all 1-Lipschitz maps from $\mathbb { R } ^ { m } \to \mathbb { R }$ . Hence, the continuity of $\mu \mapsto \mathbb { P } _ { \mu }$ transfers to $\mu \mapsto \mathbb { P } _ { n , \mu }$ . The conclusion now follows from Corollary 1 and the universality of conditional normalizing flows, cf. Lemma B.1 in Appendix B. □

## B Universality of conditional normalizing flows

To be derived, Corollary 2 requires the universality of conditional normalizing flows. To the best of our knowledge, however, such result is not readily available in the existing literature. Prior works, in fact, either focus on the unconditional case, as in [32, 35], or discuss parametric problems in terms of difeomorphism learning, as in [39], rather than in the context of conditional distributions. To bridge this gap, we derive a simple universality result, suficient for our purposes, which directly descends from [32, Theorem 1].

Lemma B.1. Let $\mathcal { P } \subset \mathbb { R } ^ { p }$ be a compact set equipped with a suitable density $\xi = \xi ( \mu )$ . Assume that each $\mu \in \mathcal P$ is associated with some probability distribution $\mathbb { P } _ { \mu }$ defined over $\mathbb { R } ^ { n }$ . Further assume that each of those probability distributions is absolutely continuous with respect to the Lebesgue measure, and that the union of their supports is bounded. Finally, assume that the map $\mu \mapsto \mathbb { P } _ { \mu }$ is continuous with respect to the $\mathcal { W } _ { 1 }$ metric. Let Q the standard Gaussian distribution in $\mathbb { R } ^ { n }$ . Then, for every $\epsilon > 0$ , there exists a conditional normalizing flow ϕ with three alternating afine coupling layers, and maps s, t represented by ReLU networks, such that

$$
\int _ { \mathcal { P } } \mathcal { W } _ { 1 } ( \phi ( \cdot , \mu ) \# \mathbb { Q } , \ \mathbb { P } _ { \mu } ) \xi ( \mu ) d \mu < \epsilon .
$$

Proof. Let $\epsilon ^ { \prime } > 0$ . By compactness and continuity, the map $\mu \mapsto \mathbb { P } _ { \mu }$ is uniformly continuous. Hence, there exists some $\delta = \delta ( \epsilon ^ { \prime } )$ for which $\mathcal { W } _ { 1 } ( \mathbb { P } _ { \pmb { \mu } } , \mathbb { P } _ { \pmb { \mu } ^ { \prime } } ) < \epsilon ^ { \prime }$ whenever $\| \pmb { \mu } - \pmb { \mu } ^ { \prime } \| < \delta .$ . Leveraging compactness, let $\{ \mu _ { 1 } , \ldots , \mu _ { m } \}$ be a δ-net for $\mathcal { P }$ . Let $\mathbb { P } _ { 1 } , \ldots , \mathbb { P } _ { m }$ be the corresponding probabilities, so that $\mathbb { P } _ { j } : = \mathbb { P } _ { \mu _ { j } }$ . In light of the universality of unconditiona normalizing flows, for each of such probabilities there exists a Real $\mathrm { N V P ~ } \phi _ { j }$ with three coupling layers, featuring ReLU networks only, such that $\mathcal { W } _ { 1 } ( \phi _ { j } \# \mathbb { Q } , \mathbb { P } _ { j } ) < \epsilon ^ { \prime } \colon$ see [32, Theorem 1] and Remark B.1.

For each $j = 1 , \ldots , m$ , let $s _ { j , 1 } , s _ { j , 2 } , s _ { j , 3 }$ and $t _ { j , 1 } , t _ { j , 2 } , t _ { j , 3 }$ be the ReLU networks in the three afine coupling layers of $\phi _ { j }$ . Define the parametric maps

$$
\widetilde { s } _ { k } ( \pmb { \mu } , \pmb { x } ) : = \sum _ { j = 1 } ^ { m } \mathbf { 1 } _ { B _ { j } } ( \pmb { \mu } ) s _ { j , k } ( \pmb { x } ) , \qquad \widetilde { t } _ { k } ( \pmb { \mu } , \pmb { x } ) : = \sum _ { j = 1 } ^ { m } \mathbf { 1 } _ { B _ { j } } ( \pmb { \mu } ) t _ { j , k } ( \pmb { x } )\tag{43}
$$

where $\mathcal { B } _ { 1 } , \ldots , B _ { m }$ is a partition obtained by considering the m balls of radius $\delta$ centered over the δ-net $\pmb { \mu } _ { 1 } , \ldots , \pmb { \mu } _ { m }$ namely $B _ { j } : = \tilde { B } _ { j } \setminus ( \cup _ { l = 1 } ^ { j - 1 } B _ { l } )$ for $j \ge 2 , \mathcal { B } _ { 1 } : = \tilde { B } _ { 1 }$ and $\tilde { B } _ { j } : = \{ \pmb { \mu } \in \mathcal { P } \ : \ \| \pmb { \mu } - \pmb { \mu } _ { j } \| < \delta \}$ . Let $\tilde { \phi }$ be the exogeneous conditional normalizing flow having $\tilde { s } _ { k }$ and $\tilde { t } _ { k }$ as maps for the alternating afine coupling layers. We call $\tilde { \phi }$ exogenoeus because its defining maps are not given by neural network architectures but rather more general functions.

Given any $\pmb { \mu } \in \mathcal { P }$ , let $j ( \mu )$ be the unique index in $\{ 1 , \ldots , m \}$ for which $\mu \in B _ { j ( \mu ) }$ . Then, for every $\pmb { \mu } \in \mathcal { P }$ we have

$$
\begin{array} { r l } & { \mathcal { W } _ { 1 } ( \widetilde { \phi } ( \cdot , \mu ) \# \mathbb { Q } , \ : \mathbb { P } _ { \mu } ) \leq \mathcal { W } _ { 1 } ( \widetilde { \phi } ( \cdot , \mu ) \# \mathbb { Q } , \ : \mathbb { P } _ { j ( \mu ) } ) + \mathcal { W } _ { 1 } \big ( \mathbb { P } _ { j ( \mu ) } , \mathbb { P } _ { \mu } \big ) } \\ & { \qquad = \mathcal { W } _ { 1 } \big ( \phi _ { j ( \mu ) } \# \mathbb { Q } , \ : \mathbb { P } _ { j ( \mu ) } \big ) + \mathcal { W } _ { 1 } \big ( \mathbb { P } _ { \mu _ { j ( \mu ) } } , \mathbb { P } _ { \mu } \big ) \leq 2 \epsilon ^ { \prime } . } \end{array}\tag{44}
$$

We now notice that, by assumption, there exists a compact set $s$ such that ${ \mathcal { S } } \supset \operatorname { s u p p } ( \mathbb { P } _ { \mu } )$ for all $\pmb { \mu } \in \mathcal { P }$ . Having fixed such S, let $\eta > 0$ be a parameter, whose value shall be chosen later on. Leveraging universality of ReLU networks, for every $k = 1 , \ldots , 3$ , let $s _ { k } ^ { \bar { \eta } }$ and $t _ { k } ^ { \eta }$ be two ReLU networks, function of $\pmb { \mu }$ and ${ \mathbf { } } ^ { \mathbf { } } \mathbf { { \mathbf { x } } } ,$ such that

$$
\int _ { \mathcal { P } \times \in S } \| \tilde { s } _ { k } ( \mu , x ) - s _ { k } ^ { \eta } ( \mu , x ) \| \xi ( \mu ) d \mu < \eta \qquad \mathrm { a n d } \qquad \int _ { \mathcal { P } ^ { x \in S } } \operatorname* { s u p } _ { | \mathcal { \tilde { F } } | \in \mathcal { P } } \| \tilde { t } _ { k } ( \mu , x ) - t _ { k } ^ { \eta } ( \mu , x ) \| \xi ( \mu ) d \mu < \eta .\tag{45}
$$

This is possible since the maps $\tilde { s } _ { k }$ and $\tilde { t } _ { k }$ are all continuous in x and piecewise constant in $\pmb { \mu } .$ . Without loss of generality, we also require that $\begin{array} { r } { \bar { | s _ { k } ^ { \eta } | } , | t _ { k } ^ { \eta } | \le M } \end{array}$ uniformly across $\mathcal { P } \times \mathcal { S }$ , with M independent of $\eta$ (but dependent on $\epsilon ^ { \prime }$ through s˜ and $\tilde { t , }$ which depend on $\phi _ { 1 } , \ldots , \phi _ { m } )$ . Such boundness can be ensured in multiple ways. For instance, one may first approximate the indicator functions in (43) using continuous maps (e.g., partitions of unity), obtaining suitable continuous approximants of $\tilde { s } _ { k }$ and $\tilde { t } _ { k }$ , which can then be approximated uniformly on $\mathcal { P } \times \mathcal { S }$ by ReLU networks.

Let now $\phi _ { \eta }$ be the conditional normalizing flow associated with $\{ s _ { k } ^ { \eta } , t _ { k } ^ { \eta } \} _ { k = 1 } ^ { 3 }$ . For every $\mu \in { \mathcal { P } }$ let $\tilde { g } _ { \mu }$ be the density of $\widetilde { \phi } ( \cdot , \pmb { \mu } ) \# \mathbb { Q }$ and let $g _ { \mu } ^ { \eta }$ be that of $\phi _ { \eta } ( \cdot , \pmb { \mu } ) \# \mathbb { Q }$ . We first notice that, by (45), for almost every $\pmb { \mu } \in \mathcal { P }$ we have $s _ { k } ^ { \eta } ( \pmb { \mu } , \cdot )  \tilde { s } _ { k } ( \pmb { \mu } , \cdot )$ and $t _ { k } ^ { \eta } ( { \pmb \mu } , \cdot )  \tilde { t } _ { k } ( { \pmb \mu } , \cdot )$ uniformly on $s$ as $\eta  0 ^ { + }$ . Because the density of a normalizing flow depends smoothly on the maps s and t in its coupling layers, it is straightforward to see that such convergence directly transfers to the conditional densities, meaning that $\begin{array} { r } { \operatorname* { s u p } _ { { \pmb x } \in S } \| g _ { \pmb \mu } ^ { \eta } ( { \pmb x } ) - \tilde { g } _ { \pmb \mu } ( { \pmb x } ) \|  0 } \end{array}$ as $\eta  0 ^ { + }$ for almost every $\pmb { \mu } \in \mathcal { P }$ . However, because all densities are uniformly bounded by some constant $C = C ( M ) > 0 .$ , this immediately implies

$$
\operatorname* { l i m } _ { \eta \to 0 ^ { + } } \int _ { \mathcal { P } } \operatorname* { s u p } _ { { \pmb x } \in { \mathscr { S } } } \| \tilde { g } _ { \pmb \mu } ( { \pmb x } ) - g _ { \pmb \mu } ^ { n } ( { \pmb x } ) \| \xi ( { \pmb \mu } ) d { \pmb \mu } = 0 ,
$$

by dominated convergence. In particular, because $s$ is bounded, we also have $\begin{array} { r } { \int _ { \mathcal { P } } \mathcal { W } _ { 1 } ( \tilde { \phi } ( \cdot , \pmb { \mu } ) \# \mathbb { Q } , \phi _ { \eta } ( \cdot , \pmb { \mu } ) \# \mathbb { Q } ) \xi ( \pmb { \mu } ) d \pmb { \mu } \to } \end{array}$ 0 as $\eta  0 ^ { + }$ . Thus, we may choose $\eta = \eta _ { * } ( \epsilon ^ { \prime } )$ such that $\begin{array} { r } { \int _ { \mathcal { P } } \mathcal { W } _ { 1 } ( \tilde { \phi } ( \cdot , \pmb { \mu } ) \# \mathbb { Q } , \ \phi _ { \eta } ( \cdot , \pmb { \mu } ) \# \mathbb { Q } ) \xi ( \pmb { \mu } ) d \pmb { \mu } < \epsilon ^ { \prime } . } \end{array}$ . Combining the latter with (44) we obtain

$$
\int _ { \mathcal { P } } \mathcal { W } _ { 1 } ( \phi _ { \eta _ { - } } ( \cdot , \mu ) \# \mathbb { Q } , \mathbb { P } _ { \mu } ) \xi ( \mu ) d \mu \leq \int _ { \mathcal { P } } \mathcal { W } _ { 1 } ( \tilde { \phi } ( \cdot , \mu ) \# \mathbb { Q } , \ \phi _ { \eta _ { + } } ( \cdot , \mu ) \# \mathbb { Q } ) \xi ( \mu ) d \mu + \int _ { \mathcal { P } } \mathcal { W } _ { 1 } ( \tilde { \phi } ( \cdot , \mu ) \# \mathbb { Q } , \mathbb { P } _ { \mu } ) \xi ( \mu ) d \mu < 3 \epsilon ^ { \prime } .
$$

Setting $\epsilon ^ { \prime } : = \epsilon / 3$ and letting $\phi : = \phi _ { \eta _ { \ast } ( \epsilon / 3 ) }$ yields the conclusion.

Remark B.1. Our definition of Real NVPs difers slightly from that in $I { \mathcal { B } } { \mathcal { Q } } J ,$ as we require a terminal activation $\rho _ { k }$ in the scaling network s of each coupling layer. Nevertheless, [32, Theorem 1] extends naturally to our setting whenever $\rho _ { k }$ is continuous and unbounded below. Indeed, the universality proof in $\it { 1 3 2 }$ uses only constant negative scalings of arbitrarily large magnitude. It therefore sufices to show that there exists $M \in \mathbb { R }$ such $t h a t ,$ for every $c < M$ , our definition admits a scaling network $s _ { c }$ satisfying $s _ { c } ( \pmb { x } ) \equiv [ c , \ldots , c ]$ for all inputs x. To this end, set $M : = \rho _ { k } ( 0 )$ . By continuity and unboundedness, it follows that $( - \infty , M ] \subseteq \rho _ { k } ( \mathbb { R } )$ . Hence, for every $c < M$ , there exists $\tau _ { c } \in \mathbb { R }$ such that $\rho _ { k } ( \tau _ { c } ) = c$ . The desired network can then obtained by setting all weights and biases to zero, except for the bias of the output layer, which can be constructed as $[ \tau _ { c } , \dots , \tau _ { c } ]$

From a practical standpoint, the widespread use of bounded activations may pose some limitations in this direction, but their incompatibility with the above assumption is not as severe as it may sound. The universality discussed so far, in fact, covers a broad class oftarget distributions, including absolutely continuous distributions with disconnected supports. Such targets cannot be represented exactly by a difeomorphic transformation of a Gaussian, since difeomorphisms preserve connectedness of the support. For this reason, their approximation generally requires transformations whose Jacobians become increasingly singular. In the construction discussed above, this behavior arises as $c \to - \infty$ , causing the corresponding scaling factors to approach zero. Bounded activations restrict access to this limiting regime, but they can still be very well-suited for approximating a broad class of diferent distributions.