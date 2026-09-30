# Improving Function Space Flow Matching with Kernel Optimal Transport

Fred Xu University of California, Los Angeles fredxu@cs.ucla.edu

Barbora Barancikova Imperial College London b.barancikova23@imperial.ac.uk

Thomas Markovich Block, Inc. tmarkovich@squareup.com

Yizhou Sun University of California, Los Angeles yzsun@cs.ucla.edu

## Abstract

Generative models for function-valued data, such as time series and solutions of partial differential equations, must learn distributions over infinite-dimensional spaces. Functional Flow Matching (FFM) extends Flow Matching to this setting, learning a velocity field whose flow transports a Gaussian prior to the data distribution, but it inherits the independent endpoint pairing of standard Flow Matching: in each batch, prior and data samples are matched arbitrarily, so the conditional bridge must traverse both the shared global structure of the dataset and instance-specific residuals. In function space this limitation is harder to fix than in finite dimensions, since optimal transport (OT) on function spaces is delicate to formulate and a flat Euclidean surrogate ignores the geometry that distinguishes function-valued data. We propose kernel Functional Flow Matching (kFFM), which replaces the independent pairing by entropic OT under a kernel-induced cost, the coupling underlying the Hilbert Sinkhorn Divergence (HSD), while leaving the FFM neural-operator architecture unchanged. Theoretically, we prove that the kernel cost and the HSD objective are uniformly bounded and well-posed on Banach ambient spaces, derive an explicit error decomposition against quadratic-cost OT on compact metric spaces that isolates an irreducible kernel-cost mismatch term, and prove a discretization-invariance bound whose rate is governed by Sobolev regularity. Empirically, kFFM improves distributional matching over FFM, diffusion, adversarial, and finite-dimensional OT baselines on time-series and PDE benchmarks, with statistically significant paired-seed gains over FFM and improvements that persist under non-kernel and physics-based diagnostics, including a turbulent Navier–Stokes benchmark. Bounded kernel costs already outperform raw $L ^ { 2 }$ Sinkhorn, and function-space-aware kernels (signature, Sobolev RBF) give further gains on rough or path-valued data.

## 1 Introduction

Flow matching [35] is a powerful framework for generative modeling that learns the velocity field of a probability path between a reference and a data distribution. Time series and PDE snapshots are naturally represented as functions, motivating Kerrigan et al. [25] to extend this framework to function space through Functional Flow Matching (FFM), which defines a path of Gaussian measures in a Hilbert space and parameterizes the velocity field with a neural operator. FFM enjoys resolution-invariance and competitive performance on sequence and PDE datasets, but inherits the independent endpoint coupling of base FM. Under this coupling, each Gaussian prior sample is paired with an arbitrary data function in a minibatch, and the conditional bridge must traverse both the shared global structure of the dataset and instance-specific residuals; the velocity field is then forced to discover the global structure and match each particular instance simultaneously. A natural fix is geometry-aware coupling: pairing each prior sample with a data sample that is already close in a meaningful geometry, so the bridge primarily transports the residual. This idea has improved finite-dimensional flow matching [48], but transferring it to function space is non-trivial: direct optimal transport (OT) on function spaces is delicate to formulate and compute [8, 9, 22, 37], and a flat $L ^ { 2 }$ surrogate ignores the function-space geometry (Sobolev structure, path roughness) that distinguishes one function-valued sample from another.

We resolve this through a kernel-induced Hilbert space surrogate. Working in a Reproducing Kernel Hilbert Space [46], the entropic coupling underlying the Hilbert Sinkhorn Divergence [31] provides a tractable pairing rule whose kernel encodes prior knowledge about function-space geometry: Sobolev structure for PDE snapshots via the RBF kernel on $H ^ { k }$ , and p-variation roughness for time series via the signature kernel [5, 6]. Our theory certifies this surrogate coupling (well-posedness with uniform bounds, an error decomposition with an irreducible kernel-mismatch term, and discretization invariance); it makes no straightness or sampling-efficiency claim in function space (Section 4, Appendix C.14), and the sample-quality gains are an empirical claim that we test with paired-seed statistics, non-kernel metrics, physics diagnostics, and a turbulent Navier–Stokes stress test. The kernel-induced surrogate also outperforms raw $L ^ { 2 }$ Sinkhorn: relative to a flat- $L ^ { 2 }$ minibatch-OT baseline (CFM-OT(L<sup>2</sup>)) with the same Sinkhorn solver and neural-operator backbone, kFFM lowers the distributional error by 75% on Gene Expression and 66% on Economy. Function-space-aware kernels (Sobolev RBF, signature) add further gains on rough or path-valued data. Code for every experiment is publicly available.<sup>1</sup> Our main contributions are the following:

1. We propose kernel Functional Flow Matching (kFFM), to our knowledge the first method to perform OT endpoint coupling in function-space flow matching: a kernel-induced cost couples prior and data samples before flow matching, while the FFM neural-operator architecture is unchanged.

2. We establish three theoretical results for the function-space setting: uniform boundedness and well-posedness of the kernel cost and HSD on Banach ambient spaces; an explicit error decomposition against quadratic-cost OT on compact metric spaces (kernel mismatch, entropic bias, covering); and a discretization-invariance theorem with an $N ^ { - 2 ( \alpha - k ) / d }$ rate that ties the function-space objective to the truncated grid representation the model actually computes.

3. We show empirically that kFFM improves population-level distributional quality on time series and PDE snapshots over functional generative baselines and a finite-dimensional OT coupling baseline, with paired-seed significance, gains that survive non-kernel and physicsbased metrics including a turbulent Navier–Stokes benchmark, and a training overhead of only 5–7% at the largest resolution we train. The coupling geometry matters differently across data classes: signature kernels for paths, RBF-type costs for fields.

## 2 Background

For the FFM construction we follow Kerrigan et al. [25] and work in a separable Hilbert space $\mathcal { F }$ of functions $f : \Omega \to { \mathbb { R } }$ , with $\Omega \subset \mathbb { R } ^ { d }$ compact, equipped with its Borel σ-algebra $B ( \mathcal { F } ) ; \bar { \nu } \in \mathcal { P } ( \mathcal { F } )$ denotes the data measure. Section 2.1 works directly with ${ \mathcal { F } } ,$ while later subsections and Section 3 use X for the metric sample space on which the OT coupling is defined. Additional background is deferred to Appendix A.

## 2.1 Flow Matching in Function Space

Flow Matching [35] learns the velocity field whose flow transports a prior distribution to the data distribution along a given probability path. In the Gaussian-measure formulation of FFM, a vector field $v : [ 0 , 1 ] \times \mathcal { F } \stackrel { \left. } { \right. } \mathcal { F }$ induces the flow $\partial _ { t } \psi _ { t } ( g ) = v _ { t } ( \psi _ { t } ( g ) ) , \psi _ { 0 } ( g ) = g .$ and the path $\mu _ { t } = [ \psi _ { t } ] _ { \# } \mu _ { 0 }$ from $\mu _ { 0 } \in { \mathcal { P } } ( { \mathcal { F } } )$ . FFM constructs this path by averaging endpoint-conditioned Gaussian bridges over $f \sim \nu ;$ if $\mu _ { 0 } ^ { f } = \mu _ { 0 }$ and $\mu _ { 1 } ^ { f }$ is concentrated around $f ,$ , then $\textstyle \mu _ { t } ( A ) = \int \mu _ { t } ^ { f } ( A ) d \nu ( f )$ satisfies $\mu _ { 1 } = \nu .$ . For Gaussian bridges with $\mu _ { 0 } = \mathcal { N } ( 0 , C _ { 0 } )$ and $\mu _ { t } ^ { f } = \mathcal { N } ( m _ { t } ^ { f } , ( \sigma _ { t } ^ { f } ) ^ { 2 } C _ { 0 } )$ , the conditional

velocity is

$$
v _ { t } ^ { f } ( g ) = \frac { ( \sigma _ { t } ^ { f } ) ^ { \prime } } { \sigma _ { t } ^ { f } } ( g - m _ { t } ^ { f } ) + \frac { d } { d t } m _ { t } ^ { f } , \qquad g \sim \mu _ { t } ^ { f } ,
$$

and the standard FFM parametrization $m _ { t } ^ { f } = t f , \sigma _ { t } ^ { f } = 1 - ( 1 - \sigma _ { \operatorname* { m i n } } ) t$ corresponds to Gaussian OT between the bridge marginals, while the endpoint pairing itself remains independent [48]. A neural operator $v _ { \theta } ( g , t )$ , here a Fourier Neural Operator (FNO) [32], is trained with the Conditional Flow Matching (CFM) loss

$$
\begin{array} { r } { \mathcal { L } _ { C F M } ( \theta ) : = \mathbb { E } _ { t \sim U [ 0 , 1 ] , f \sim \nu , g \sim \mu _ { t } ^ { f } } \left[ \| v _ { t } ^ { f } ( g ) - v _ { \theta } ( g , t ) \| _ { \mathcal { F } } ^ { 2 } \right] , } \end{array}\tag{1}
$$

which under suitable regularity conditions [25, 35] has the same minimizers as the unconditional loss $\mathcal { L } _ { F M } ( \theta ) : = \mathbb { E } _ { t \sim U [ 0 , 1 ] , g \sim \mu _ { t } } [ \| v _ { t } ( g ) - v _ { \theta } ( g , t ) \| _ { \mathcal { F } } ^ { 2 } ]$ ; the absolute-continuity assumption behind this equivalence is spelled out in Appendix B.2. In Section 3 we keep the same probability path but replace the independent pairing of its endpoints by an OT-coupled pairing.

## 2.2 Optimal Transport and Kernel-based Optimal Transport

For $( \mathcal { X } , d )$ a Polish space, $\mu , \nu \in \mathcal P ( \mathcal X )$ , and a cost c, the static OT problem is $\mathcal { W } ( \mu , \nu ) : =$ $\operatorname { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int$ c dπ [49], and its entropic regularization [38]

$$
W _ { \epsilon } ( \mu , \nu ) : = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int c ( x , y ) d \pi ( x , y ) + \epsilon \mathrm { K L } ( \pi \| \mu \otimes \nu )\tag{2}
$$

underlies Sinkhorn methods; the debiased Sinkhorn divergence subtracts the corresponding self-costs [16, 17, 47]. OT-based couplings improve finite-dimensional flow matching by computing a static coupling between reference and data samples before interpolation [28, 48]. Adapting this idea to function-valued data is non-trivial: no Lebesgue-like reference measure exists on function space [11, 37], so densities are ill-posed, and Gaussian-measure and Cameron–Martin assumptions [8] are limiting when data are non-Gaussian (PDE states, rough paths). This motivates embedding the measures into a Reproducing Kernel Hilbert Space (RKHS) [46], where the coupling is computed under a kernel-induced cost.

Definition 1 (Hilbert Sinkhorn Divergence [31]). Let $( \mathcal { X } , d )$ be a metric space, $\mu , \nu \in { \mathcal { P } } ( { \mathcal { X } } )$ , and $\kappa : \mathcal { X } \times \mathcal { X }  \mathbb { R }$ a measurable symmetric positive-definite kernel with RKHS ${ \mathcal { H } } _ { \kappa }$ and canonical feature map $\phi : \mathcal { X }  \mathcal { H } _ { \kappa }$ . The entropic OT functional on the embedded space is

$$
\begin{array} { r } { \mathsf { O T } _ { \epsilon } ^ { \mathcal { H } _ { \kappa } } \bigl ( \phi _ { \# } \mu , \phi _ { \# } \nu \bigr ) : = \operatorname* { i n f } _ { \substack { \pi _ { \phi } \in \Pi ( \phi _ { \# } \mu , \phi _ { \# } \nu ) } } \Big [ \int \| u - v \| _ { \mathcal { H } _ { \kappa } } ^ { 2 } d \pi _ { \phi } ( u , v ) + \epsilon \mathrm { K L } \big ( \pi _ { \phi } \| \phi _ { \# } \mu \otimes \phi _ { \# } \nu ) \Big ] , } \end{array}\tag{3}
$$

and the Hilbert Sinkhorn divergence is the debiased quantity $S _ { \epsilon } ( \phi _ { \# } \mu , \phi _ { \# } \nu ) : = \mathsf { O T } _ { \epsilon } ^ { \mathcal { H } _ { \kappa } } ( \phi _ { \# } \mu , \phi _ { \# } \nu ) -$ $\textstyle \frac { 1 } { 2 } { \mathsf { O T } } _ { \epsilon } ^ { \mathcal { H } _ { \kappa } } \bigl ( \phi _ { \# } \mu , \phi _ { \# } \mu \bigr ) - \frac { 1 } { 2 } { \mathsf { O T } } _ { \epsilon } ^ { \mathcal { H } _ { \kappa } } \bigl ( \phi _ { \# } \nu , \phi _ { \# } \nu \bigr )$

By the kernel trick, the functional in Equation (3) can be written directly on X with the RKHS-induced cost $c _ { \kappa } ( x , y ) : = \| \phi ( x ) - \phi ( y ) \| _ { \mathcal { H } _ { \kappa } } ^ { 2 } = \overline { { \kappa } } ( x , x ) + \kappa ( y , y ) - 2 \kappa ( x , y ) [ 3 1 $ , restated as Proposition 3 in Appendix B]:

$$
\mathsf { O T } _ { \kappa , \epsilon } ( \mu , \nu ) : = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \Big [ \int _ { \mathcal { X } \times \mathcal { X } } c _ { \kappa } ( x , y ) d \pi ( x , y ) + \epsilon \mathrm { K L } ( \pi \| \mu \otimes \nu ) \Big ] ,\tag{4}
$$

with minimizers corresponding under pushforward, and $\begin{array} { r } { S _ { \kappa , \epsilon } ( \mu , \nu ) : = 0 \mathsf { T } _ { \kappa , \epsilon } ( \mu , \nu ) - \frac { 1 } { 2 } \mathsf { O } \mathsf { T } _ { \kappa , \epsilon } ( \mu , \mu ) - } \end{array}$ $\textstyle { \frac { 1 } { 2 } } \boldsymbol { \mathrm { O } } \mathsf { T } _ { \kappa , \epsilon } ( \nu , \nu ) = S _ { \epsilon } ( \phi _ { \# } \mu , \phi _ { \# } \nu )$ . Previous work takes $\mathcal { X } \subset \mathbb { R } ^ { d }$ , where common kernels apply directly, the divergence enjoys statistical consistency with finite-sample bounds (restated in Appendix B), and the discrete entropic OT problem is convex [16]. Section 3 generalizes X to function-valued sample spaces.

## 2.3 Path Signature and Signature Kernel

Let $C _ { p } (  { \mathbb { R } } ^ { d } ) = C _ { p } ( [ 0 , T ] ,  { \mathbb { R } } ^ { d } )$ denote the continuous paths from $[ 0 , T ]$ to $\mathbb { R } ^ { d }$ with finite p-variation, $p \in [ 1 , 2 )$ . The signature of $x \in C _ { p } ( \mathbb { R } ^ { d } )$ is the sequence of iterated Young integrals $S ( x ) =$ $( 1 , S ^ { ( 1 ) } ( x ) , S ^ { ( 2 ) } ( x ) , \ldots )$ with $\begin{array} { r } { { S } ^ { ( m ) } ( x ) = \int _ { 0 < t _ { 1 } < \cdots < t _ { m } < T } d x _ { t _ { 1 } } \otimes \cdots \cdot \cdot \otimes d x _ { t _ { m } } \in ( \mathbb { R } ^ { d } ) ^ { \otimes m } } \end{array}$ , an element of the free tensor algebra $\begin{array} { r } { T ( ( \mathbb { R } ^ { d } ) ) = \prod _ { n = 0 } ^ { \infty } ( \dot { \mathbb { R } ^ { d } } ) ^ { \otimes n } } \end{array}$

Definition 2 (Signature Kernel [5]). The signature kernel $\kappa _ { \mathrm { s i g ~ } } : \ : C _ { p } ( \mathbb { R } ^ { d } ) \times \ : C _ { p } ( \mathbb { R } ^ { d } ) \to $ R is $\begin{array} { r } { \kappa _ { \mathrm { s i g } } ( x , y ) = \langle S ( x ) , S ( y ) \rangle : = \sum _ { m = 0 } ^ { \infty } \langle S ^ { ( m ) } ( x ) , S ^ { ( m ) } ( y ) \rangle _ { ( \mathbb { R } ^ { d } ) ^ { \otimes m } } } \end{array}$ , where $\langle \cdot , \cdot \rangle _ { ( \mathbb { R } ^ { d } ) \otimes m }$ is the Hilbert– Schmidt inner product.

The signature kernel is symmetric and positive definite and hence induces a unique RKHS $\mathcal { H } _ { \mathrm { s i g } } ;$ on compact subsets, the time-augmented signature kernel is universal and characteristic [5]. We compute it with pySigLib [44].

## 3 Functional Flow Matching with Kernel Optimal Transport

Throughout this section X is a general metric space, the sample space on which data and reference measures live and on which the kernel OT coupling acts.

## 3.1 Coupling with Kernel Optimal Transport

kFFM first couples prior and data samples through entropic OT under a kernel-induced cost, Equation (4). The kernel reflects the metric space where the data live. On the Sobolev space $\mathcal { X } = \dot { H } ^ { k } ( \Omega )$ (functions with k weak derivatives in $L ^ { 2 }$ , the natural home of PDE solutions [12]), the RBF kernel on $H ^ { k }$ rewards pairs close in a derivative-aware norm; on path space $\mathcal { X } \subset C _ { p } ( [ 0 , T ] , \mathbb { R } ^ { d } ) , p \in [ 1 , 2 )$ (smooth and rough time series [5]), the signature kernel rewards reparametrization-invariant path similarity. Well-posedness for each choice is established in Section 3.2 and validated in Section 4; with empirical measures the minibatch problem is solved by the Sinkhorn algorithm [17].

Given a plan $\pi$ between prior draws $f _ { 0 } \sim \mu _ { 0 }$ and data samples $f _ { 1 } \sim \mu _ { 1 }$ , kFFM uses the same probability path as FFM, with the paired prior sample in the role of the base noise:

$$
\begin{array} { r l } & { \quad g _ { t } ^ { f _ { 0 } , f _ { 1 } } = t f _ { 1 } + \sigma _ { t } f _ { 0 } , \qquad \sigma _ { t } = 1 - ( 1 - \sigma _ { \operatorname* { m i n } } ) t , \qquad ( f _ { 0 } , f _ { 1 } ) \sim \pi , } \\ & { \quad v _ { t } ^ { f _ { 0 } , f _ { 1 } } \left( g _ { t } ^ { f _ { 0 } , f _ { 1 } } \right) = \partial _ { t } g _ { t } ^ { f _ { 0 } , f _ { 1 } } = f _ { 1 } - ( 1 - \sigma _ { \operatorname* { m i n } } ) f _ { 0 } = \frac { f _ { 1 } - \left( 1 - \sigma _ { \operatorname* { m i n } } \right) g _ { t } ^ { f _ { 0 } , f _ { 1 } } } { \sigma _ { t } } , } \end{array}
$$

so that $g _ { 0 } = f _ { 0 } \sim \mu _ { 0 } , g _ { 1 } = f _ { 1 } + \sigma _ { \operatorname* { m i n } } f _ { 0 }$ , and the conditional velocity is the FFM conditional field of Section 2.1 evaluated with the coupled pair; the FFM path $t f + \sigma _ { t } \xi$ is recovered exactly for the independent coupling $\pi = \mu _ { 0 } \otimes \mu _ { 1 }$

What the coupling changes. Up to the $\sigma _ { \mathrm { m i n } }$ term, the regression target is the displacement $u : =$ $f _ { 1 } - f _ { 0 }$ , the residual between the paired data and prior samples, which carries all the endpoint information. Under independent coupling this residual is the difference of two unrelated draws, a large and high-variance regression target; under kernel-OT coupling each $f _ { 1 }$ is paired with a geometrically close $f _ { 0 } ,$ , so the network regresses smaller, structured residuals. The neural operator (an FNO) is unchanged relative to FFM; only the minibatch coupling, and through it the law of the conditional paths, is modified. Algorithm 1 summarizes training; implementation defaults and good practices are collected in Appendix C.8. The reference distribution $\mu _ { 0 }$ is a Gaussian process (GP) or Gaussian random field [19], or white noise on the grid where the data are rougher than a smooth GP prior (the base measure is part of the selected configuration, Section 4.1); in practice both measures are replaced by their empirical estimators, with the coupling computed from a kernel-induced cost matrix [17, 31]:

$$
\mu _ { n } = \sum _ { i = 1 } ^ { n } \hat { \mu } _ { i } \delta _ { x _ { i } } , \qquad \nu _ { n } = \sum _ { j = 1 } ^ { n } \hat { \nu } _ { j } \delta _ { y _ { j } } , \qquad C _ { i j } ^ { \kappa } = \kappa ( x _ { i } , x _ { i } ) + \kappa ( y _ { j } , y _ { j } ) - 2 \kappa ( x _ { i } , y _ { j } ) .\tag{5}
$$

Where the kernel enters. The kernel enters at exactly one place: the cost $c _ { \kappa }$ used by minibatch Sinkhorn to pair prior and data samples. No sample or model input is projected into the RKHS; the FNO acts on the original discretized functions, the paths are constructed in the original space, and generation happens there. The training coupling is the entropic plan induced by $c _ { \kappa } ,$ , computed per minibatch with log-domain Sinkhorn on a median-normalized cost and sampled from the plan (Appendix C.8); the debiasing self-terms of the HSD (Definition 1) do not depend on π and cannot change the argmin plan, so HSD is our analysis object (what Theorems 1–2 control), not the training objective. Finally, at a fixed batch size b minibatch OT induces the expected minibatch plan $\bar { \pi } _ { b } = \mathbb { E } [ \hat { \pi } _ { b } ]$ [13, 14], a valid coupling of the true marginals for every b, so the CFM consistency argument applies and b only changes which conditional paths are used; as $b  \infty , \bar { \pi } _ { b }$ converges to the population entropic kernel-OT plan.

Algorithm 1 Functional Flow Matching with Kernel Optimal Transport   
Require: Prior distribution $\mu _ { 0 } = \mathcal { N } ( 0 , C _ { 0 } )$ , data distribution $\mu _ { 1 }$ , kernel function $\kappa ,$ entropy regular  
ization $\epsilon > 0 ,$ , batch size $b ,$ initial network v , $\sigma _ { \operatorname* { m i n } } > 0$   
1: Training loop (repeat until convergence):   
2: Sample batches of size b: $f _ { 0 } \sim \mu _ { 0 } , f _ { 1 } \sim \mu _ { 1 }$   
3: Solve the entropic kernel OT problem in Equation (4) with $\kappa , \epsilon$ (cost matrix in Equation (5))   
to get coupling π.   
4: $( f _ { 0 } , f _ { 1 } ) \sim \pi$   
5: $t \sim \mathcal { U } ( 0 , 1 ) , \quad \sigma _ { t } \gets 1 - ( 1 - \sigma _ { \operatorname* { m i n } } ) t$   
6: $g \gets t f _ { 1 } + \sigma _ { t } f _ { 0 }$ // the paired prior sample plays the role ofthe base noise   
7: $v _ { t } ^ { f _ { 0 } , f _ { 1 } } ( g )  ( f _ { 1 } - ( 1 - \sigma _ { \operatorname* { m i n } } ) g ) / \sigma _ { t } \ = \ f _ { 1 } - ( 1 - \sigma _ { \operatorname* { m i n } } ) f _ { 0 }$   
8: $\mathcal { L } _ { C F M } ( \theta )  \| v _ { t } ^ { f _ { 0 } , f _ { 1 } } ( g ) - v _ { \theta } ( g , t ) \| _ { \mathcal { F } } ^ { 2 }$   
9: $\theta \gets \mathrm { U p d a t e } ( \ddot { \theta } , \nabla _ { \theta } \mathcal { L } _ { \mathrm { C F M } } ( \theta ) )$   
10: Output: $v _ { \theta } .$

## 3.2 Theoretical Results for Sobolev and Path Space

The theory certifies the surrogate coupling; it does not by itself explain why sample quality improves, which Section 4 establishes empirically. It answers three questions (proofs in Appendix B): is the kernel-OT objective well-defined and well-conditioned on the function spaces used by kFFM; how far is the RKHS-induced cost from quadratic-cost OT, and which parts of the gap are reducible; and does the continuum formulation survive the Fourier truncation used by the FNO and the Sinkhorn solver?

Theorem 1 (Well-posedness and uniform boundedness). Let $( { \mathcal { X } } , \| \cdot \| _ { \mathcal { X } } )$ be a real Banach space and let $d _ { \mathcal { X } } ( f , g ) \ \overset { \cdot } { : = } \ \| f - g \| _ { \mathcal { X } }$ . Let $\mu , \nu \in \mathcal P ( \mathcal X )$ and let $\kappa : \mathcal { X } \times \mathcal { X } $ R be measurable, symmetric, and positive definite with RKHS $\mathcal { H } _ { \kappa } .$ . Assume additionally that κ is bounded: $B _ { \kappa } : =$ $\textstyle \operatorname* { s u p } _ { f \in { \mathcal { X } } } \kappa ( f , f ) < \infty$ . Then the kernel cost satisfies $0 \le c _ { \kappa } ( f , g ) \le 4 B _ { \kappa }$ for all $f , g \in { \mathcal { X } } ,$ , and for every $\varepsilon > 0$ the Hilbert Sinkhorn divergence $S _ { \kappa , \varepsilon } ( \mu , \nu )$ (Definition 1) isfinitefor all $\mu , \nu \in \mathcal { P } ( \mathcal { X } )$ , with $| S _ { \kappa , \varepsilon } ( \mu , \nu ) | \leq 8 B _ { \kappa }$

Finiteness per se is not the binding constraint: under a second-moment assumption on $\nu$ the $L ^ { 2 }$ cost is already integrable, and for the Gaussian prior $\mathbb { E } \| f _ { 0 } \| ^ { 2 } = \mathrm { t r } ( C _ { 0 } ) < \infty$ . Theorem 1 delivers more. (a) Uniform boundedness with no assumption on the measures, which conditions the entropic solver: the Gibbs kernel $e ^ { - c _ { \kappa } / \varepsilon }$ stays bounded away from 0 and 1 across batches, so Sinkhorn converges independently of outlier pairs, unlike unbounded $L ^ { 2 }$ costs on heavy-tailed batches (Section 4.3 quantifies the resulting speed-up). (b) Generality: the statement holds on Banach spaces without Hilbert structure, which licenses the signature kernel on path space, our strongest configuration. (c) Geometry injection: the kernel is the interface through which domain structure enters the coupling. For Sobolev data in $H ^ { k } ( \Omega )$ we use $\kappa _ { \mathrm { R B F } } ^ { ( k ) } ( x , y ) : = \exp ( - \| x - y \| _ { H ^ { k } } ^ { 2 } / 2 \sigma ^ { 2 } )$ , which is bounded and positive definite [7, 54]; for paths we use the signature kernel (Definition 2), universal and characteristic on compact sets [5], bounded there by continuity, and used for two-sample tests on path data [6], applied to time-augmented paths (Appendix A, Definition 14) to avoid its tree-like equivalence.

The entropic OT functional transports with the cost induced by the feature map rather than the original metric cost. On compact subsets of the ambient space $( B _ { R } ^ { H ^ { \alpha } }$ is compact in $H ^ { k }$ for $\alpha > k ;$ compact sets of paths for the signature kernel, with truncation discussed in Appendix B), the next theorem separates the resulting kernel-cost mismatch from entropic and covering effects.

Theorem 2 (Error decomposition). Let $( \mathcal { X } , d )$ be a compact metric space and let $\mu , \nu \in { \mathcal { P } } ( { \mathcal { X } } )$ Define the target quadratic cost $c ( x , y ) = d ( x , y ) ^ { 2 }$ . Let $\kappa : \mathcal { X } \times \mathcal { X } $ R be a bounded positivedefinite kernel with RKHS-induced cost $c _ { \kappa } ( x , y ) : = \kappa ( x , x ) + \kappa ( y , y ) - 2 \kappa ( x , y )$ , and let $\Delta _ { \kappa } : =$ $\begin{array} { r } { \operatorname* { s u p } _ { x , y \in \mathcal { X } } | c _ { \kappa } ( x , y ) - c ( x , y ) | } \end{array}$ . Let ${ \mathsf { O T } } _ { \kappa , \varepsilon }$ denote the entropic OTfunctional in Equation (4) with cost $c _ { \kappa } ,$ and let M := diam $( \mathcal { X } ) < \infty$ . Then for every $\varepsilon > 0$ and every $\delta \in ( 0 , M ] .$

$$
| 0 \mathsf T _ { \kappa , \varepsilon } ( \mu , \nu ) - \mathcal W ( \mu , \nu ) | \leq \Delta _ { \kappa } + 2 \varepsilon \log \mathcal N ( \mathcal X , \delta ; d ) + 8 M \delta ,\tag{6}
$$

where $\mathcal { N } ( \mathcal { X } , \delta ; d )$ is the δ-covering number ofX in the metric d.

Theorem 2 is an error decomposition, not an approximation guarantee: the entropic and covering terms vanish as $\varepsilon , \delta \to 0$ , but $\Delta _ { \kappa }$ is irreducible and does not shrink with more samples or finer grids. For the RBF kernel, $c _ { \mathrm { R B F } } = 2 ( 1 - e ^ { - \| f - g \| ^ { 2 } / 2 \sigma ^ { 2 } } )$ is a rescaled $L ^ { 2 }$ cost up to relative error $O ( ( D / \sigma ) ^ { 2 } )$ , with D the batch diameter; with median-normalized costs $\Delta _ { \kappa }$ is small at large bandwidth, and on smooth Navier–Stokes fields the induced plans agree across costs (total variation $\leq 0 . 0 2 ;$ Appendix C.12). For the signature kernel $\Delta _ { \kappa }$ is large by design: the theorem then controls the deviation from Wasserstein with the chosen cost, not proximity to $\overline { { L ^ { 2 } \mathrm { O T } } } .$ and the pair rankings differ measurably from $L ^ { 2 }$ (rank correlation 0.33–0.37 on AEMET and gene expression), which is where the largest gains occur. In low-regularity regimes the bound degrades on two fronts: $\Delta _ { \kappa }$ grows because a smooth kernel cost saturates on rough functions, and the covering number blows up as the effective smoothness approaches the embedding threshold, making the entropic term large for any practical ε. The benchmarks of Section 4.1 sit in the smooth regime where the bound is informative; Section 4.2 stress-tests the turbulent regime empirically.

Theorems 1 and 2 are stated at the continuum level, while the model operates on a finite-rank spectral representation: PDE snapshots are represented in a Fourier subspace of dimension proportional to $N$ which both the FNO and the Sinkhorn solver consume. Our final theorem quantifies the resulting gap at a rate governed by Sobolev regularity; the path-space analogue for signature truncation is discussed in Appendix B.

Theorem 3 (Discretization invariance). Let $\Omega \subset \mathbb { R } ^ { d }$ be a boundedperiodic domain and let $x > k \ge 0$ Set $\mathcal { X } : = B _ { R } ^ { H ^ { \alpha } } : = \{ f \in H ^ { \alpha } ( \Omega ) : \| f \| _ { H ^ { \alpha } } \leq R \}$ , viewed as a compact subset of $H ^ { k } ( \Omega )$ , and let $\mu , \nu \in \mathcal P ( \mathcal X )$ . For the Fourier basis $\{ e _ { n } \} _ { n \in \mathbb { Z } ^ { d } }$ , let $V _ { N } : = \mathrm { s p a n } \{ e _ { n } : | n | ^ { 2 } \leq N ^ { 2 / d } \}$ and let $P _ { N } : H ^ { \alpha } ( \Omega ) \to V _ { N }$ denote the orthogonal projection; hence dim $V _ { N } \asymp N .$ . Write $\mu _ { N } : = ( P _ { N } ) _ { \# } \mu$ and $\begin{array} { r } { \nu _ { N } : = ( P _ { N } ) _ { \# } \nu . } \end{array}$ . Define the Sobolev-RBF kernel $\kappa _ { \mathrm { R B F } } ^ { ( k ) } ( f , g ) : = \exp ( - \| f - g \| _ { H ^ { k } } ^ { 2 } / 2 \sigma ^ { 2 } )$ on $H ^ { k }$ and write KO $\Gamma _ { \varepsilon } ^ { \left( k \right) } : = 0 \mathsf { T } _ { \kappa _ { \mathrm { R B F } } ^ { \left( k \right) } , \varepsilon } f o r$ the corresponding entropic kernel-OT functional from Theorem 2. Then for every $\varepsilon > 0 ,$

$$
\big | { \mathsf { K O T } } _ { \varepsilon } ^ { ( k ) } ( \mu , \nu ) - { \mathsf { K O T } } _ { \varepsilon } ^ { ( k ) } ( \mu _ { N } , \nu _ { N } ) \big | \ \leq \ \frac { 4 R ^ { 2 } } { \sigma ^ { 2 } } N ^ { - 2 ( \alpha - k ) / d } .
$$

Equivalently, in terms of grid spacing $h = N ^ { - 1 / d }$ , the discretization error decays as $O ( h ^ { 2 ( \alpha - k ) } )$

Combining Theorems 2 and 3 bounds the gap between the computed finite-rank kernel-OT objective and continuum quadratic OT in the $H ^ { k }$ metric by four interpretable terms: kernel-cost mismatch, entropic regularization, covering, and discretization (Corollary 1 in Appendix B.6). Only the discretization term vanishes with projection rank $N$ , at rate $N ^ { - 2 ( \alpha - k ) / \dot { d } }$ driven by the regularity gap $\alpha - k$ , the formal counterpart of the empirical resolution-invariance in Section $4 ;$ the mismatch term depends only on kernel design (zero for the linear kernel, finite for RBF and controlled by data scale and bandwidth σ). The theorem relies on compact Sobolev embedding [12]; our smooth PDE benchmarks sit in that regime [4].

## 4 Experiments

Datasets. We use two classes of data. Sequence datasets: AEMET [15], gene expression [36], and the economics dataset [3] from Kerrigan et al. [25], plus financial time series from the Heston stochastic volatility model [21], treated as realizations of measures on path space $C _ { p } ( \mathbb { R } )$ . PDE snapshots: the 1D Korteweg–de Vries (KdV) equation, the incompressible 2D Navier–Stokes equation [33], and their stochastic counterparts [43], treated as measures on Sobolev space $H ^ { k } ( \Omega )$ ; Section 4.2 adds a turbulent 2D Navier–Stokes benchmark.

Baselines. Denoising Diffusion Operator (DDO) with NCSN noise scale [34], functional DDPM [24], Generative Adversarial Neural Operator (GANO) [40], and the original FFM [25]. We also include a direct finite-dimensional OT baseline, CFM-OT(L<sup>2</sup>) [48], which uses the same minibatch Sinkhorn solver and FNO backbone as our method but applies entropic OT to flattened grid vectors with the unbounded $L ^ { 2 }$ cost and the white-noise base of finite-dimensional flow matching; kFFM’s Euclidean– RBF configuration instead uses the bounded RBF cost on Euclidean distances. kFFM variants are named by cost, kFFM-Sig (signature), kFFM-Sob (Sobolev-RBF), kFFM-RBF (Euclidean–RBF), kFFM-Euc (raw $L ^ { 2 } )$ , and tagged by base measure, gp (GP prior) or wn (white noise). Datasets, architectures, and training budgets, identical across methods, are detailed in Appendix C.

Table 1: Population-level distributional quality measured by MMD-RBF (lower is better; unbiased estimator, so small negative values occur). “kFFM (selected)” is the per-dataset configuration chosen on a held-out validation split by sliced Wasserstein distance (Appendix C.5); each cell is tagged (cost/base) with Sig = signature kernel, RBF = Euclidean–RBF kernel, Sob-RBF = Sobolev-RBF kernel, gp = Gaussian-process prior, wn = white noise. The kFFM and FFM columns are the matchedseed runs of Table 2 (n as given there); the remaining baselines are averaged over 10 seeds. “Gain $( \% ) ^ { \dag }$ is relative to the strongest non-kFFM baseline in the row and is omitted (–) where that baseline sits at the noise floor of the unbiased estimator (non-positive MMD), where ratios are not meaningful; Table 2 reports paired tests instead.
<table><tr><td>Dataset</td><td>kFFM (selected)</td><td>CFM-OT(L2)</td><td>FFM</td><td>DDPM</td><td>DDO/NCSN</td><td>GANO</td><td>Gain (%)</td></tr><tr><td>Heston</td><td>kFFM (RBF/wn): -  $- 1 . 3 0 \times 1 0 ^ { - 4 }$ </td><td> $\phantom { 0 } { - 8 . 2 6 } \times 1 0 ^ { - 5 }$ </td><td> $3 . 1 0 \times 1 0 ^ { - 4 }$ </td><td> $1 . 1 0 \times 1 0 ^ { - 3 }$ </td><td> $5 . 5 3 \times 1 0 ^ { - 2 }$ </td><td> $3 . 9 2 \times 1 0 ^ { - 2 }$ </td><td>1</td></tr><tr><td>AEMET</td><td>kFFM (Sig/gp): –  $5 . 1 0 \times 1 0 ^ { - 3 }$ </td><td> $\cdot 4 . 1 1 \times 1 0 ^ { - 3 }$ </td><td> $- 3 . 2 4 \times 1 0 ^ { - 3 }$ </td><td> $- 3 . 9 9 \times 1 0 ^ { - 3 }$ </td><td> $2 . 4 4 \times 1 0 ^ { - 1 }$ </td><td> $5 . 2 4 \times 1 0 ^ { - 1 }$ </td><td>7</td></tr><tr><td>KdV</td><td>kFFM (Sob-RBF/gp): –  $- 2 . 5 6 \times 1 0 ^ { - 3 }$ </td><td> $\cdot 1 . 3 2 \times 1 0 ^ { - 3 }$ </td><td> $- 1 . 1 6 \times 1 0 ^ { - 3 }$ </td><td> $4 . 8 4 \times 1 0 ^ { - 2 }$ </td><td> $2 . 3 7 \times 1 0 ^ { - 1 }$ </td><td> $4 . 1 2 \times 1 0 ^ { - 1 }$ </td><td>一</td></tr><tr><td>Economy</td><td> $\mathbf { k F F M _ { \delta } ( S i g / g p ) } \colon 2 . 8 9 \times 1 0 ^ { - 3 }$ </td><td> $8 . 4 6 \times 1 0 ^ { - 3 }$ </td><td> $7 . 3 0 \times 1 0 ^ { - 3 }$ </td><td> $9 . 9 7 \times 1 0 ^ { - 3 }$ </td><td> $1 . 5 7 \times 1 0 ^ { - 1 }$ </td><td> $8 . 9 8 \times 1 0 ^ { - 1 }$ </td><td>60.4</td></tr><tr><td>Gene Expr.</td><td> $\mathbf { k F F M _ { \delta } ( S i g / g p ) } \colon 7 . 1 1 \times 1 0 ^ { - 3 }$ </td><td> $2 . 8 0 \times 1 0 ^ { - 2 }$ </td><td> $1 . 7 8 \times 1 0 ^ { - 2 }$ </td><td> $5 . 3 2 \times 1 0 ^ { - 2 }$ </td><td> $3 . 4 2 \times 1 0 ^ { - 2 }$ </td><td> $4 . 1 7 \times 1 0 ^ { - 1 }$ </td><td>60.1</td></tr><tr><td>Stoch. KdV</td><td> $\mathbf { k F F M _ { \delta } ( S o b { \bar { - } } \bar { R B F / } g p ) } ; 1 . 0 0 \times 1 0 ^ { - 4 }$ </td><td> $6 . 5 8 \times 1 0 ^ { - 4 }$ </td><td> $8 . 8 0 \times 1 0 ^ { - 4 }$ </td><td> $3 . 8 4 \times 1 0 ^ { - 4 }$ </td><td> $4 . 4 8 \times 1 0 ^ { - 2 }$ </td><td> $2 . 3 9 \times 1 0 ^ { - 2 }$ </td><td>74.0</td></tr><tr><td>Stoch. NS</td><td> $\mathbf { k F F M } ( \mathbf { R B F / w n } ) ; 2 . 8 0 \times 1 0 ^ { - 3 }$ </td><td> $4 . 2 2 \times 1 0 ^ { - 3 }$ </td><td> $9 . 6 8 \times 1 0 ^ { - 2 }$ </td><td> $1 . 2 0 \times 1 0 ^ { - 1 }$ </td><td> $2 . 9 5 \times 1 0 ^ { - 1 }$ </td><td> $2 . 5 2 \times 1 0 ^ { - 1 }$ </td><td>33.6</td></tr><tr><td>Navier-Stokes</td><td> $\mathbf { k F F M } ( \mathbf { R B F / w n } ) ; 1 . 6 3 \times 1 0 ^ { - 3 }$ </td><td> $1 . 7 8 \times 1 0 ^ { - 3 }$ </td><td> $1 . 3 1 \times 1 0 ^ { - 1 }$ </td><td> $3 . 8 0 \times 1 0 ^ { - 1 }$ </td><td> $7 . 3 5 \times 1 0 ^ { - 1 }$ </td><td> $3 . 0 0 \times 1 0 ^ { - 1 }$ </td><td>8.4</td></tr></table>

Metrics. The maximum mean discrepancy with an RBF kernel (MMD-RBF) is the primary population-level metric, computed with the unbiased estimator (so small negative values occur). To avoid judging an RBF-type coupling by an RBF-MMD alone, we also report non-kernel metrics (sliced Wasserstein distance and marginal $W _ { 1 } )$ , the pointwise diagnostics of Kerrigan et al. [25] (autocorrelation error on sequences, log-spectral distance on PDEs, mean/variance summaries in Appendix C), and, on Navier–Stokes, physics diagnostics that capture non-Gaussian structure: enstrophy-distribution $W _ { 1 }$ , vorticity-PDF $W _ { 1 }$ , skewness and excess-kurtosis errors of the pooled vorticity, and log-spectral error. Results aggregate 10 seeds (20 where marked); all methods share seeds, so tests are paired by seed.

## 4.1 Distributional Quality

Table 1 reports MMD-RBF against all baseline families. The “kFFM (selected)” column reports, per dataset, the configuration (OT cost and Gaussian base measure) selected on a held-out validation split by sliced Wasserstein distance, a non-kernel criterion (Appendix C.5); pointwise diagnostics by kernel variant appear in Appendix C.6. kFFM is the strongest method on every row, and the selected configurations follow a practitioner-usable pattern: the signature kernel with a GP base on path-space datasets, where function-space geometry matters most (except short-horizon Heston, where the top configurations are statistically indistinguishable and the rule picks the bounded RBF cost with a white-noise base); the Sobolev-RBF cost with a GP base on the KdV family; and the bounded Euclidean–RBF cost with a white-noise base on the Navier–Stokes family, whose rough fields are poorly matched by a smooth GP prior.

Uncertainty and paired tests. Since unbiased MMD values can be tiny or negative, percentage gains are uninformative near the noise floor. Table 2 therefore reports per-seed mean±std and seed-paired tests against FFM at matched architecture and training budget: kFFM improves on FFM on every dataset at level 0.05 under both a paired t-test and a Wilcoxon signed-rank test, and wins on every shared seed for the four largest-gap datasets. In the GP-base rows only the coupling changes; in the white-noise rows (Heston, Navier–Stokes family) the base measure changes too, and at that base the effect of the kernel cost is isolated by the comparison against CFM- $. \mathrm { O T } ( \mathbf { \breve { L } } ^ { 2 } )$ in Table 1.

Non-kernel metrics. The gains survive the non-kernel metrics (Table 3): on the same runs, kFFM improves sliced Wasserstein distance and marginal $W _ { 1 }$ over FFM on every dataset, by 18–20% on Gene Expression and Economy, where each comparison is significant when paired by seed, and by 2.9–4.1× on Navier–Stokes and Stoch. NS; for the signature configurations the coupling and evaluation kernels already differ. The pointwise diagnostics also expose what MMD understates: $\mathrm { C F M - O T } ( L ^ { 2 } )$ buys its pointwise-marginal fit with 105× and 66× worse autocorrelation error than kFFM-Sig on Economy and Gene Expression (Table 9); function-level structure is where the kernel coupling is strongest.

Bounded cost or function-space geometry? Both effects are real and dominate on different data classes. On smooth fields the RBF coupling is a monotone transform of the $L ^ { 2 }$ cost (rank correlation ≈ 1 on training batches; Appendix C.12), so it reuses the $L ^ { 2 }$ geometry while compressing outliers, which buys conditioning, ϵ-robustness, and faster Sinkhorn, and matches or improves the unbounded $L ^ { 2 }$ coupling at the same base measure (Stoch. $\mathrm { N S } \colon 2 . 8 0 ~ \mathrm { v s } . 4 . 2 2 \times 1 0 ^ { - 3 }$ , Table 1). On paths, no Euclidean cost, raw or bounded, recovers the signature gains (matched runs, $\mathbf { M M D \times } 1 0 ^ { 3 } ;$ Gene Expr.: kFFM-Euc 20.0, kFFM-RBF 20.7 vs. kFFM-Sig 7.1; Economy: 7.55/7.60 vs. 2.9), and the signature kernel genuinely re-ranks pairs (rank correlation 0.33–0.37 with $L ^ { 2 } ) ;$ a synthetic example in Appendix C.13 isolates both failure modes of $L ^ { 2 }$ pairing. Qualitative samples and super-resolution results are shown in Appendix C (Figures 2 and 3).

Table 2: kFFM (selected) vs. FFM with uncertainty, at matched architecture and training budget. $\mathbf { M M D - R B F } { \times } 1 0 ^ { 3 }$ as mean±std over n shared seeds (fixed before outcomes); p-values from a seedpaired t-test and a Wilcoxon signed-rank test. In the gp rows only the coupling changes at a fixed prior; in the wn rows the selected kFFM also uses a white-noise base whereas FFM keeps its GP prior (at that base, the comparison isolating the kernel cost is against $\mathrm { C F M - O T } ( L ^ { 2 } )$ in Table 1). Economy averages the population and GDP series within each seed. kFFM wins on every shared seed for the first four rows.
<table><tr><td>Dataset</td><td>n</td><td>FFM</td><td>kFFM (selected)</td><td>paired t p-value</td><td>Wilcoxon p-value</td></tr><tr><td>Gene Expr.</td><td>10</td><td> $1 7 . 8 \pm 3 . 4$ </td><td> ${ \bf 7 . 1 1 \pm 2 . 0 \ ( S i g / g p ) }$ </td><td> $2 . 0 \times 1 0 ^ { - 5 }$ </td><td>0.002</td></tr><tr><td>Economy</td><td>10</td><td> $7 . 3 0 \pm 1 . 8$ </td><td> ${ \bf 2 . 8 9 \pm 0 . 9 7 ( S i g / g p ) }$ </td><td> $2 . 5 \times 1 0 ^ { - 5 }$ </td><td>0.002</td></tr><tr><td>Stoch. NS</td><td>10</td><td> $9 6 . 8 \pm 2 1$ </td><td> $\mathbf { 2 . 8 0 \pm 1 . 3 ~ ( R B F / w n ) }$ </td><td> $1 . 7 \times 1 0 ^ { - 7 }$ </td><td>0.002</td></tr><tr><td>Navier-Stokes</td><td>10</td><td> $1 3 1 \pm 6 0$ </td><td> ${ \bf 1 . 6 3 \pm 2 . 6 \ ( R B F / w n ) }$ </td><td> $8 . 6 \times 1 0 ^ { - 5 }$ </td><td>0.002</td></tr><tr><td>AEMET</td><td>20</td><td> $- 3 . 2 4 \pm 2 . 1$ </td><td> ${ \bf - 5 . 1 0 \pm 2 . 6 ( S i g / g p ) }$ </td><td>0.015</td><td>0.014</td></tr><tr><td>Heston</td><td>20</td><td> $0 . 3 1 \pm 0 . 8 0$ </td><td> $\mathbf { - 0 . 1 3 \pm 0 . 3 0 ( R \bar { B } \bar { F } / w n ) }$ </td><td>0.021</td><td>0.011</td></tr><tr><td>KdV</td><td>20</td><td> $- 1 . 1 6 \pm 2 . 1$ </td><td> $\mathbf { - 2 . 5 6 \pm 1 . 0 ( S o b . R B F / g p ) }$ </td><td>0.039</td><td>0.027</td></tr><tr><td>Stoch. KdV</td><td>20</td><td> $0 . 8 8 \pm 1 . 1$ </td><td> $\mathbf { 0 . 1 0 \pm 0 . 3 3 ( S o b . R B F / g p ) }$ </td><td>0.037</td><td>0.037</td></tr></table>

Table 3: Non-kernel metrics for kFFM (selected configuration in parentheses) vs. FFM on the runs of Table $2 ;$ lower is better. Sequence rows: mean±std over 10 shared seeds (Economy averages the population and GDP series within each seed), each comparison significant when paired by seed (Wilcoxon $p = 0 . 0 0 2 )$ ; Navier–Stokes rows are means. Appendix Table 16 adds the autocorrelation and log-spectrum errors.
<table><tr><td></td><td colspan="2">Sliced-W</td><td colspan="2">Marginal-W1</td></tr><tr><td>Dataset</td><td>FFM</td><td>kFFM</td><td>FFM</td><td>kFFM</td></tr><tr><td>Gene Expr. (Sig/gp)</td><td> $0 . 1 0 6 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 0 8 7 \pm 0 . 0 0 6 }$ </td><td> $0 . 0 7 4 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 0 5 9 \pm 0 . 0 0 4 }$ </td></tr><tr><td>Economy  $( { \mathrm { S i g } } / { \mathrm { g p } } )$ </td><td> $0 . 0 2 9 1 \pm 0 . 0 0 1 9$ </td><td>0.0233 ± 0.0012</td><td> $0 . 0 1 7 1 \pm 0 . 0 0 1 6$ </td><td> $\mathbf { 0 . 0 1 3 7 \pm 0 . 0 0 0 9 }$ </td></tr><tr><td>Navier-Stokes (RBF/wn)</td><td>0.315</td><td>0.081</td><td>0.239</td><td>0.058</td></tr><tr><td>Stoch. NS (RBF/wn)</td><td>1.06</td><td>0.365</td><td>0.795</td><td>0.249</td></tr></table>

Kernel choice and sensitivity. Re-aggregating the one-at-a-time sweeps of Appendix C.16: results are indistinguishable within seed-to-seed variation for RBF bandwidths $\sigma \in [ 0 . 1 , 2 ]$ on every dataset, flat across three orders of magnitude of the Sinkhorn regularization $( \epsilon \in [ 0 . 0 0 1 , 1 ] )$ for bounded costs (the raw $L ^ { 2 }$ cost degrades by ≈ 10× at $\epsilon \geq 0 . 5$ on long volatility paths), and unchanged across the signature hyperparameters (dyadic order $0 { - } 3 ,$ , static bandwidth 0.1–5); over batch sizes $b \in \{ 6 4 , \ldots , 5 1 2 \}$ , kFFM tracks FFM across the 8× range (Appendix C.8). The kernel family is the only choice that matters, and the rule we recommend is simple and consistent with the theory: match the kernel to the data class (signature for path-like data; Sobolev or Euclidean RBF for smooth fields, where the Euclidean RBF is a reasonable default; RBF or $L ^ { 2 }$ on very long paths, where signature-kernel memory grows with path length), set the bandwidth by the median heuristic, and validate with a non-kernel metric on a held-out split (Appendix C.5). The properties the theorems require (boundedness, characteristicness) hold for every kernel we consider, so the theory constrains the family only weakly and the selection rule does the rest.

## 4.2 Turbulent Regime and Physics Diagnostics

The Navier–Stokes benchmark above $( \nu = 1 0 ^ { - 3 } )$ is smooth, the regime where our theory is informative. To test the low-regularity regime where Theorem 2 is weakest, we run the canonical turbulent

Table 4: Turbulent 2D Navier–Stokes $( \nu = 1 0 ^ { - 5 }$ , 1200 trajectories, $\mathrm { R e } \approx 2 0 0 0 ) $ : distributional and physics diagnostics, lower is better, mean±std over 5 seeds with identical FNO backbone and training budget. Rows marked <sup>†</sup> use the GP base measure; kFFM and CFM-OT use the white-noise base chosen by the validation rule (Appendix C.5). Best per column in bold (on MMD, kFFM and CFM-OT tie within std). Unscaled values and full configurations are in Appendix Table 17.
<table><tr><td>Method</td><td>MMD  $( \times 1 0 ^ { - 3 } )$ </td><td>Sliced-W</td><td> $( \times 1 0 ^ { - 2 } )$ </td><td>Vort-PDF W1 Enstrophy W1</td><td>Skew err.  $( \times 1 0 ^ { - 2 } )$ </td><td>Kurt err.  $( \times 1 0 ^ { - 2 } )$ </td><td> $\mathrm { l o g - S p e c }$   $( \times \mathrm { { 1 0 } ^ { - 2 } } )$ </td></tr><tr><td>kFFM (selected)</td><td> ${ \bf 1 . 4 8 \pm 0 . 6 }$ </td><td> ${ \bf 0 . 1 9 3 \pm 0 . 0 1 }$ </td><td> ${ \bf 3 . 8 \pm 1 }$ </td><td> ${ \bf 0 . 1 5 5 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 2 7 \pm 0 . 3 }$ </td><td> $7 . 8 \pm 1$ </td><td> ${ \bf 2 . 0 \pm 0 . 8 }$ </td></tr><tr><td>CFM-OT(L2)</td><td> ${ \bf 1 . 4 7 \pm 1 . 0 }$ </td><td> $0 . 1 9 4 \pm 0 . 0 1$ </td><td> $4 . 1 \pm 1$ </td><td> $0 . 1 5 7 \pm 0 . 0 1$ </td><td> $0 . 4 0 \pm 0 . 2$ </td><td> $7 . 5 \pm 1$ </td><td> $2 . 2 \pm 0 . 7$ </td></tr><tr><td>kFFM-Sob†</td><td> $1 1 1 \pm 2 0$ </td><td> $0 . 6 1 2 \pm 0 . 0 5$ </td><td> $1 0 . 4 \pm 3$ </td><td> $0 . 1 9 4 \pm 0 . 0 5$ </td><td> $4 . 6 \pm 2$ </td><td> ${ \bf 6 . 8 \pm 5 }$ </td><td> $6 . 5 \pm 1$ </td></tr><tr><td>kFFM-Euc†</td><td> $1 0 9 \pm 2 0$ </td><td> $0 . 6 2 3 \pm 0 . 0 3$ </td><td> $1 0 . 4 \pm 4$ </td><td> $0 . 1 9 7 \pm 0 . 0 6$ </td><td> $5 . 0 \pm 3$ </td><td> $7 . 7 \pm 4$ </td><td> $6 . 7 \pm 1$ </td></tr><tr><td>FFM†</td><td> $1 0 8 \pm 2 0$ </td><td> $0 . 6 0 1 \pm 0 . 0 2$ </td><td> $1 0 . 7 \pm 4$ </td><td> $0 . 2 0 1 \pm 0 . 0 6$ </td><td> $4 . 9 \pm 3$ </td><td> $7 . 2 \pm 5$ </td><td> $6 . 5 \pm 1$ </td></tr><tr><td>DDO/NCSN</td><td> $5 6 9 \pm 9$ </td><td> $1 . 3 9 \pm 0 . 0 6$ </td><td> $1 2 2 \pm 1$ </td><td> $1 . 0 7 \pm 0 . 0 2$ </td><td> $0 . 8 3 \pm 1$ </td><td> $1 2 9 \pm 1$ </td><td> $9 3 . 7 \pm 2 $ </td></tr><tr><td>DDPM</td><td> $3 8 0 \pm 2 0$ </td><td> $1 . 1 8 \pm 0 . 1$ </td><td> $8 4 . 8 \pm 3$ </td><td> $0 . 8 8 9 \pm 0 . 0 2$ </td><td> $7 . 9 \pm 5$ </td><td> $3 3 7 \pm 2 0 0$ </td><td> $1 1 . 7 \pm 3$ </td></tr></table>

Turbulent 2D NS $( \nu = 1 0 ^ { - 5 } ) ;$ distributional diagnostics, seed 1

![](images/9197dc502f70456d13cac969fb3a5f25039f6a05182379a526372dfd11cca60a.jpg)

![](images/3fd3dc986bf1288f59b7cb8e5360796056e7698f371cdca5719f8932439a9282.jpg)  
Figure 1: Turbulent 2D Navier–Stokes $( \nu = 1 0 ^ { - 5 }$ , seed 1): pooled vorticity PDF (log scale) and radially averaged energy spectrum. The data PDF is bimodal and heavy-tailed; kFFM (selected) tracks it, DDO/NCSN collapses to a narrow near-Gaussian spike, and DDPM is unimodal with the wrong shape. The flow-matching spectra track the data across scales; DDO/NCSN’s is nearly flat.

FNO benchmark [32]: 2D Navier–Stokes at $\nu = 1 0 ^ { - 5 }$ (1200 trajectories, from which we take the spun-up snapshots at $t \geq 1 0 ;$ Re ≈ 2000; seven methods, five seeds, identical FNO backbone and training budget). Table 4 reports MMD, sliced-W, and the physics diagnostics; Figure 1 shows the pooled vorticity PDF and energy spectrum, and vorticity samples are in Appendix C.11. (i) kFFM (selected) attains the best value on five of seven metrics and ties the best on MMD; held-out sliced-W rejects the GP-base arms (<sup>†</sup> rows; 0.19 vs. ≥ 0.60), and among the tied white-noise arms the rule keeps the Euclidean-RBF configuration selected on standard NS; CFM- $\mathrm { \cdot O T } ( L ^ { 2 } )$ at the same base overlaps kFFM within std, with means favoring kFFM on five of seven metrics. (ii) OT-coupled flow matching is far more robust than the score-based and diffusion alternatives: 23–32× better vorticity- $\cdot \mathrm { P D F } \mathbf { \bar { W } } _ { 1 } , 5 . 7 \mathbf { - } 6 . 9 \times$ better enstrophy $W _ { 1 } .$ , and 17–43× better vorticity-kurtosis error (Figure 1). (iii) Against FFM run with its GP prior, kFFM improves every metric except kurtosis (a tie), with seed-paired significance (MMD 73× lower, $p = \overset { \cdot } { 3 } \times 1 0 ^ { - 4 }$ ; sliced-W $3 . 1 \times , p = 1 . 3 \times 1 0 ^ { - 6 } ;$ log-spectrum 3.2×, p = 0.005; vorticity-PDF 2.9×, $p = 0 . 0 0 7 ;$ ; skewness $1 8 \times , p = 0 . 0 1 6 )$ . This gain combines the base-measure change and the coupling: the three GP-base arms are indistinguishable, so in this regime the white-noise base does most of the work, and at that base the kernel and $\mathrm { r a w } { - \cal L } ^ { 2 }$ costs tie, as in (i). All methods, ours included, still leave headroom on sliced-W in this regime (Section 6).

## 4.3 Cost, Efficiency, and Convergence

Compute. Measured on every benchmark (Appendix C.9), the end-to-end training overhead of the per-minibatch Sinkhorn solve over FFM is 5–7% at the largest resolution we train $( 1 2 8 ^ { 2 }$ Navier– Stokes); the $O ( b ^ { 2 } )$ coupling cost is independent of model size and resolution, so its relative weight shrinks as the backbone grows. Peak coupling memory is 80–151 MB (1.5–9% of the FNO’s forwardplus-backward peak), and per-batch coupling time is comparable to one FNO step. The bounded RBF cost is about $8 \times$ faster than unbounded- $L ^ { 2 }$ Sinkhorn at $b = 5 1 2$ and trains 15–25% faster end-to-end, the conditioning effect of Theorem 1; giving FFM the extra budget as additional epochs does not close the gap.

No straightness claim. We claim no straightness theorem: Appendix C.14 (Proposition 6, Corollary 2) shows that kFFM changes only the endpoint coupling while the affine interpolation template is fixed, and Hilbert-space Gaussian geometry does not inherit the Euclidean straightness picture that motivates finite-dimensional OT-CFM [52]. We therefore make no efficiency claim; Appendix C.15 reports only the training-time convergence of the target metric, where kFFM ends training at a better MMD-RBF than FFM on every dataset.

## 5 Related Work

Minibatch OT coupling for flow matching was introduced by Tong et al. [48] and Pooladian et al. [39], with follow-ups refining the transport objective [28, 51] and analyses of minibatch plans [13, 14]; all assume $\mathbb { R } ^ { \hat { d } }$ sample spaces. In function space, FFM [25] lifts flow matching to Hilbert spaces via Gaussian measures, alongside functional diffusion [24, 34] and adversarial neural operators [40]. Concurrent function-space flow matching changes the objective, the task, or the prior, but not the coupling: Functional Rectified Flow [53] straightens flows by re-training on model-generated endpoint pairs across training rounds, a mechanism complementary to and composable with our within-batch, data-geometry coupling; Operator Flow Matching [30] and flow matching with GP priors [27] target conditional time-series forecasting with an independent prior-data coupling. Kernel embeddings of measures [20, 46] underlie MMD and two-sample testing, recently on path space with the signature kernel [6]; Sinkhorn divergences interpolate between OT and MMD [16], and the Hilbert Sinkhorn Divergence [31] kernelizes them. To our knowledge, ours is the first work to perform OT endpoint coupling in function-space flow matching; Appendix D gives an extended discussion, including MMD gradient flows.

## 6 Conclusion and Limitations

kFFM replaces the independent endpoint pairing of FFM with a geometry-aware kernel OT coupling while keeping the neural-operator backbone. Theoretically, the kernel-OT surrogate is uniformly bounded and well-posed on Banach spaces, admits an explicit error decomposition against quadraticcost OT that isolates an irreducible kernel-cost mismatch, and is discretization-invariant at rate $N ^ { - 2 ( \alpha - k ) / d }$ . Empirically, kFFM improves distributional matching on time-series and PDE benchmarks with paired-seed significance, and the gains survive non-kernel metrics, physics diagnostics, and a turbulent Navier–Stokes stress test, at a training overhead of 5–7% at the largest resolution we train.

Limitations. Several deserve plain statement. First, our theory concerns the surrogate coupling: it certifies well-posedness, an error decomposition, and discretization invariance, but says nothing about straightness or sampling efficiency in function space, on which we make no claim, and it does not by itself explain the sample-quality gains, which remain an empirical finding. Second, the theory is least informative in low-regularity regimes: on the turbulent benchmark the prior-data $L ^ { 2 }$ distances more than double relative to the standard one (median 91 vs. 40), pushing the $\mathrm { R B F }$ cost into saturation, where $\Delta _ { \kappa }$ grows and the covering term of Theorem 2 is large for any practical $\epsilon ;$ there the empirical results carry the method. Third, turbulent small-scale statistics remain an open problem for function-space generative models. On current evidence the failure sits mainly with the score-based and diffusion baselines (vorticity-kurtosis errors of 1.3–3.4 and enstrophy $W _ { 1 } \approx 1 )$ , but all methods, ours included, leave headroom on sliced-W (0.19 for the best method), and in this regime the white-noise base rather than the coupling does most of the work; OT-coupled flow matching is currently the strongest option. Finally, signature-kernel memory grows with path length (0.63 GB at 100 time points vs. 12.5 GB at 512), so very long paths fall back to RBF or $L ^ { 2 }$ costs. Future work includes stronger function-space OT formulations [28, 52] and base measures and costs designed for rough, turbulent data.

## Acknowledgments and Disclosure of Funding

Barbora Barancikova is supported by UK Research and Innovation [UKRI Centre for Doctoral Training in AI for Healthcare grant number EP/S023283/1]. Fred Xu is partially supported by NSF grant 2531008 and was supported by Block, Inc. during an internship.

## References

[1] Michael Arbel, Anna Korba, Adil Salim, and Arthur Gretton. Maximum mean discrepancy gradient flow. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/ hash/944a5ae3483ed5c1e10bbccb7942a279-Abstract.html.

[2] Barbora Barancikova, Zhuoyue Huang, and Cristopher Salvi. SigDiffusions: Score-based diffusion models for time series via log-signature embeddings. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=Y8KK9kjgIK.

[3] Jutta Bolt and Jan Luiten van Zanden. Maddison style estimates of the evolution of the world economy. A new 2020 update. Maddison-Project Working Paper WP-15, The Maddison Project, October 2020. URL https://www.rug.nl/ggdc/historicaldevelopment/maddison/ publications/wp15.pdf.

[4] Guher Camliyurt, Igor Kukavica, and Vlad Vicol. Analyticity up to the boundary for the Stokes and the Navier–Stokes systems. Transactions ofthe American Mathematical Society, 373(5): 3375–3422, 2020. doi: 10.1090/tran/7990. URL https://doi.org/10.1090/tran/7990.

[5] Thomas Cass and Cristopher Salvi. Lecture notes on rough paths and applications to machine learning, 2024. URL https://arxiv.org/abs/2404.06583.

[6] Ilya Chevyrev and Harald Oberhauser. Signature moments to characterize laws of stochastic processes. Journal of Machine Learning Research, 23(176):1–42, 2022. URL https://jmlr. org/papers/v23/20-1466.html.

[7] Andreas Christmann and Ingo Steinwart. Universal kernels on non-standard input spaces. In J. Lafferty, C. Williams, J. Shawe-Taylor, R. Zemel, and A. Culotta, editors, Advances in Neural Information Processing Systems, volume 23. Curran Associates, Inc., 2010. URL https://proceedings.neurips.cc/paper\_files/paper/2010/file/ 4e0cb6fb5fb446d1c92ede2ed8780188-Paper.pdf.

[8] Giuseppe Da Prato. An Introduction to Infinite-Dimensional Analysis. Universitext. Springer Berlin Heidelberg, 1st edition, 2006. ISBN 978-3-540-29020-9. doi: 10.1007/3-540-29021-4. URL https://link.springer.com/book/10.1007/3-540-29021-4.

[9] Vincent Divol, Jonathan Niles-Weed, and Aram-Alexandre Pooladian. Optimal transport map estimation in general function spaces. The Annals of Statistics, 53(3):963–988, 2025. doi: 10.1214/24-AOS2482. URL https://doi.org/10.1214/24-AOS2482.

[10] R. M. Dudley. Real Analysis and Probability, volume 74 of Cambridge Studies in Advanced Mathematics. Cambridge University Press, Cambridge, 2nd edition, 2002. ISBN 978-0-521-00754-2. URL https://www.cambridge.org/core/books/ real-analysis-and-probability/26DDF2D09E526185F2347AA5658B96F6.

[11] Nathaniel Eldredge. Analysis and probability on infinite-dimensional spaces, 2016. URL https://arxiv.org/abs/1607.03591.

[12] Lawrence C. Evans. Partial Differential Equations, volume 19 of Graduate Studies in Mathematics. American Mathematical Society, 2nd edition, 2010. ISBN 978-0-8218-4974-3. URL https://bookstore.ams.org/gsm-19-r.

[13] Kilian Fatras, Younes Zine, Rémi Flamary, Rémi Gribonval, and Nicolas Courty. Learning with minibatch Wasserstein: asymptotic and gradient properties. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2020.

[14] Kilian Fatras, Younes Zine, Szymon Majewski, Rémi Flamary, Rémi Gribonval, and Nicolas Courty. Minibatch optimal transport distances; analysis and applications. arXiv preprint arXiv:2101.01792, 2021.

[15] Manuel Febrero-Bande and Manuel Oviedo de la Fuente. Statistical computing in functional data analysis: The R package fda.usc. Journal of Statistical Software, 51(4):1–28, 2012. doi: 10.18637/jss.v051.i04. URL https://www.jstatsoft.org/article/view/v051i04.

[16] Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé, and Gabriel Peyré. Interpolating between optimal transport and MMD using Sinkhorn divergences. In Kamalika Chaudhuri and Masashi Sugiyama, editors, Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pages 2681–2690. PMLR, 2019. URL https://proceedings. mlr.press/v89/feydy19a.html.

[17] Rémi Flamary, Nicolas Courty, Alexandre Gramfort, Mokhtar Z. Alaya, Aurélie Boisbunon, Stanislas Chambon, Laetitia Chapel, Adrien Corenflos, Kilian Fatras, Nemo Fournier, Léo Gautheron, Nathalie T. H. Gayraud, Hicham Janati, Alain Rakotomamonjy, Ievgen Redko, Antoine Rolet, Antony Schutz, Vivien Seguy, Danica J. Sutherland, Romain Tavenard, Alexander Tong, and Titouan Vayer. POT: Python optimal transport. Journal ofMachine Learning Research, 22(78):1–8, 2021. URL https://jmlr.org/papers/v22/20-451.html.

[18] Alexandre Galashov, Valentin De Bortoli, and Arthur Gretton. Deep MMD gradient flow without adversarial training. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=Pf85K2wtz8.

[19] Jacob R. Gardner, Geoff Pleiss, Kilian Q. Weinberger, David Bindel, and Andrew Gordon Wilson. GPyTorch: Blackbox matrix-matrix Gaussian process inference with GPU acceleration. In Advances in Neural Information Processing Systems, volume 31, 2018. URL https://proceedings.neurips.cc/paper/2018/hash/ 27e8e17134dd7083b050476733207ea1-Abstract.html.

[20] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alexander J. Smola. A kernel method for the two-sample-problem. In Advances in Neural Information Processing Systems 19, pages 513–520, 2006. URL https://proceedings.neurips.cc/paper\_files/paper/2006/hash/ e9fb2eda3d9c55a0d89c98d6c54b5b3e-Abstract.html.

[21] Steven L. Heston. A closed-form solution for options with stochastic volatility with applications to bond and currency options. The Review of Financial Studies, 6(2):327–343, April 1993. doi: 10.1093/rfs/6.2.327. URL https://doi.org/10.1093/rfs/6.2.327.

[22] Bamdad Hosseini, Alexander W. Hsu, and Amirhossein Taghvaei. Conditional optimal transport on function spaces. SIAM/ASA Journal on Uncertainty Quantification, 13(1):304–338, 2025. doi: 10.1137/23M1618922. URL https://epubs.siam.org/doi/10.1137/23M1618922.

[23] Zacharia Issa, Blanka Horvath, Maud Lemercier, and Cristopher Salvi. Non-adversarial training of neural SDEs with signature kernel scores. In Advances in Neural Information Processing Systems, volume 36, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/ 2023/hash/2460396f2d0d421885997dd1612ac56b-Abstract-Conference.html.

[24] Gavin Kerrigan, Justin Ley, and Padhraic Smyth. Diffusion generative models in infinite dimensions. In Francisco Ruiz, Jennifer Dy, and Jan-Willem van de Meent, editors, Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 9538–9563. PMLR, April 2023. URL https://proceedings.mlr.press/v206/kerrigan23a.html.

[25] Gavin Kerrigan, Giosue Migliorini, and Padhraic Smyth. Functional flow matching. In Sanjoy Dasgupta, Stephan Mandt, and Yingzhen Li, editors, Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 3934–3942. PMLR, May 2024. URL https://proceedings.mlr. press/v238/kerrigan24a.html.

[26] Patrick Kidger, James Morrill, James Foster, and Terry Lyons. Neural controlled differential equations for irregular time series. In Advances in Neural Information Processing Systems, volume 33, pages 6696–6707, 2020. URL https://proceedings.neurips.cc/paper/ 2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html.

[27] Marcel Kollovieh, Marten Lienen, David Lüdke, Leo Schwinn, and Stephan Günnemann. Flow matching with Gaussian process priors for probabilistic time series forecasting. In International Conference on Learning Representations (ICLR), 2025. URL https://openreview.net/ forum?id=UJ5WggV1xG.

[28] Nikita Kornilov, Petr Mokrov, Alexander Gasnikov, and Alexander Korotin. Optimal flow matching: Learning straight trajectories in just one step. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/ 079017-3310. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ hash/bc8f76d9caadd48f77025b1c889d2e2d-Abstract-Conference.html.

[29] Michel Ledoux and Michel Talagrand. Probability in Banach Spaces: Isoperimetry and Processes, volume 23 of Ergebnisse der Mathematik und ihrer Grenzgebiete (3). Springer-Verlag, Berlin, Heidelberg, 1991. URL https://link.springer.com/book/10.1007/ 978-3-642-20212-4.

[30] Yolanne Yi Ran Lee and Kyriakos Flouris. Operator flow matching for timeseries forecasting, 2025. URL https://arxiv.org/abs/2510.15101.

[31] Qian Li, Zhichao Wang, Gang Li, Jun Pang, and Guandong Xu. Hilbert Sinkhorn divergence for optimal transport. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3835–3844, June 2021. URL https://openaccess.thecvf.com/content/CVPR2021/html/Li\_ Hilbert\_Sinkhorn\_Divergence\_for\_Optimal\_Transport\_CVPR\_2021\_paper.html.

[32] Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhat tacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. In The Ninth International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=c8P9NQVtmnO.

[33] Zongyi Li, Miguel Liu-Schiaffini, Nikola B. Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew M. Stuart, and Anima Anandkumar. Learning chaotic dynamics in dissipative systems. In Advances in Neural Information Processing Systems, volume 35, pages 16768–16781, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/hash/6ad68277e27b42c60ac228c9859fc1a2-Abstract-Conference.html.

[34] Jae Hyun Lim, Nikola B. Kovachki, Ricardo Baptista, Christopher Beckham, Kamyar Azizzadenesheli, Jean Kossaifi, Vikram Voleti, Jiaming Song, Karsten Kreis, Jan Kautz, Christopher Pal, Arash Vahdat, and Anima Anandkumar. Score-Based diffusion models in function space. Journal of Machine Learning Research, 26(158):1–62, 2025. URL https://jmlr.org/papers/v26/23-1472.html.

[35] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

[36] David A. Orlando, Charles Y. Lin, Allister Bernard, Jean Y. Wang, Joshua E. S. Socolar, Edwin S. Iversen, Alexander J. Hartemink, and Steven B. Haase. Global control of cell-cycle transcription by coupled CDK and network oscillators. Nature, 453(7197):944–947, June 2008. doi: 10.1038/nature06955. URL https://www.nature.com/articles/nature06955.

[37] John C. Oxtoby. Invariant measures in groups which are not locally compact. Transactions of the American Mathematical Society, 60(2):215–237, September 1946. ISSN 0002- 9947. doi: 10.1090/S0002-9947-1946-0018188-5. URL https://doi.org/10.1090/ S0002-9947-1946-0018188-5.

[38] Gabriel Peyré and Marco Cuturi. Computational optimal transport with applications to data sciences. Foundations and Trends® in Machine Learning, 11(5–6):355–607, 2019. doi: 10.1561/2200000073. URL https://doi.org/10.1561/2200000073.

[39] Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, and Ricky T. Q. Chen. Multisample flow matching: Straightening flows with minibatch couplings. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 28100–28127. PMLR, 2023. URL https://proceedings.mlr.press/v202/pooladian23a.html.

[40] Md Ashiqur Rahman, Manuel A. Florez, Anima Anandkumar, Zachary E. Ross, and Kamyar Azizzadenesheli. Generative adversarial neural operators. Transactions on Machine Learning Research, October 2022. ISSN 2835-8856. URL https://openreview.net/forum?id= X1VzbBU6xZ.

[41] Walter Rudin. Functional Analysis. McGraw-Hill Series in Higher Mathematics. McGraw-Hill, New York, 1973. ISBN 978-0-07-054225-9. URL https://books.google.com/books/ about/Functional\_Analysis.html?id=BB\_vAAAAMAAJ.

[42] Cristopher Salvi, Thomas Cass, James Foster, Terry Lyons, and Weixin Yang. The signature kernel is the solution of a Goursat PDE. SIAM Journal on Mathematics of Data Science, 3 (3):873–899, 2021. doi: 10.1137/20M1366794. URL https://epubs.siam.org/doi/10. 1137/20M1366794.

[43] Cristopher Salvi, Maud Lemercier, and Andris Gerasimovics. Neural stochastic PDEs: Resolution-invariant learning of continuous spatiotemporal dynamics. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ 091166620a04a289c555f411d8899049-Abstract-Conference.html.

[44] Daniil Shmelev and Cristopher Salvi. pySigLib – fast signature-based computations on CPU and GPU, 2025. URL https://arxiv.org/abs/2509.10613.

[45] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In The Ninth International Conference on Learning Representations, 2021. URL https: //openreview.net/forum?id=PxTIG12RRHS.

[46] Bharath K. Sriperumbudur, Arthur Gretton, Kenji Fukumizu, Bernhard Schölkopf, and Gert R. G. Lanckriet. Hilbert space embeddings and metrics on probability measures. Journal of Machine Learning Research, 11(50):1517–1561, 2010. URL https://jmlr.org/papers/ v11/sriperumbudur10a.html.

[47] Thibault Séjourné, Jean Feydy, François-Xavier Vialard, Alain Trouvé, and Gabriel Peyré. Sinkhorn divergences for unbalanced optimal transport, 2019. URL https://arxiv.org/ abs/1910.12958.

[48] Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. Transactions on Machine Learning Research, pages 1–34, March 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=CD9Snc73AW.

[49] Cédric Villani. Optimal Transport: Old and New, volume 338 of Grundlehren der mathematischen Wissenschaften. Springer, Berlin, Heidelberg, 2009. doi: 10.1007/978-3-540-71050-9. URL https://link.springer.com/book/10.1007/978-3-540-71050-9.

[50] Yasuo Yamasaki. Measures on Infinite Dimensional Spaces, volume 5 of Series in pure mathematics. World Scientific, 1985. ISBN 9789971978525. URL https://books.google. com/books?id=F243SflQYbQC.

[51] Angxiao Yue, Anqi Dong, and Hongteng Xu. OAT-FM: Optimal acceleration transport for improved flow matching, 2025. URL https://arxiv.org/abs/2509.24936.

[52] Ho Yun and Yoav Zemel. Gaussian optimal transport beyond Brenier’s theorem, 2025. URL https://arxiv.org/abs/2512.21464.

[53] Jianxin Zhang and Clayton Scott. Flow straight and fast in hilbert space: Functional rectified flow, 2025. URL https://arxiv.org/abs/2509.10384.

[54] Johanna Ziegel, David Ginsbourger, and Lutz Dümbgen. Characteristic kernels on Hilbert spaces, Banach spaces, and on sets of measures, 2022. URL https://arxiv.org/abs/2206.07588.

## A Additional Technical Background

This appendix collects the mathematical background on which the paper relies: probability measures on function spaces, reproducing kernel Hilbert spaces and kernel embeddings, path signatures, and optimal transport.

## A.1 Probability Measure in Function Space

In this section we provide the definitions and necessary theoretical results for probability measure in function space, highlighting the analytical difficulties in general. We give high-level intuition and omit the proofs; for details we refer the reader to the textbooks of Rudin [41] and Da Prato [8].

Definition 3 (Hilbert Space). A Hilbert space H is an inner-product space that is complete in the norm induced by its inner product, defined as $\langle \cdot , \cdot \rangle : \mathcal { H } \times \mathcal { H } \to \mathbb { F }$ , where $\mathbb { F } = \mathbb { R }$ for real Hilbert space.

Importantly, the Hilbert space’s inner product structure gives it a metric space with the norm $| | x | | : =$ $\sqrt { \langle x , x \rangle }$ and metric $d ( x , y ) = | | x - y | | , \quad \forall x , y \in \mathcal { H }$ . Let $B ( x , r )$ be the open balls defined using the metric $d ,$ then the Borel σ-algebra of $\mathcal { H } .$ denoted $B ( \mathcal { H } )$ , is generated by the open balls defined in this way:

$$
\begin{array} { r } { B ( \mathcal { H } ) : = \sigma ( \{ B ( x , r ) : x \in \mathcal { H } , r > 0 \} ) . } \end{array}
$$

Another desirable property of the Hilbert space is eigenfunction spectral decomposition for bounded linear operators:

Theorem 4. Let $T : { \mathcal { H } } \to { \mathcal { H } }$ be a compact, self-adjoint, bounded linear operator. Then there exists an orthonormal set of eigenvectors $\{ e _ { n } \} _ { n \in I }$ and real eigenvalues $\{ \lambda _ { n } \} _ { n \in I }$ such that:

1. $T e _ { n } = \lambda _ { n } e _ { n }$

2. $\lambda _ { n }  0 i f I$ is infinite.

3. $\lambda _ { i } \neq 0$ has finite multiplicity.

4. The closure of span of non-zero eigenvectors produces closure of Ran(T).

5. The spectral decomposition of T is:

$$
T x = \sum _ { i \in I } \lambda _ { i } \langle x , e _ { i } \rangle e _ { i } , \qquad x \in \mathcal { H }
$$

In the main text, we adopt a measure-theoretic view of probability, which is crucial for function spaces. This is due to the following result, known as “no Lebesgue measure in infinite dimensions.” The statement usually takes the form: If X is an infinite-dimensional separable normed linear space, then there is no nontrivial, translation-invariant, σ-additive Borel measure $\mu$ on $\mathcal { X }$ that is finite on every nonempty open ball [50]. The implication of this result is that there is no universal definition of probability density as in $\mathbb { R } ^ { d }$ , and only the more general Radon-Nikodym derivative can be defined between two measures that are absolutely continuous.

Definition 4 (Singular and Absolute Continuity). For a measurable space $( \Omega , { \mathcal { F } } )$ , two measures $\mu , \nu$ are mutually singular, written $\mu \perp \nu ,$ , if there exists $A \in { \mathcal { F } } .$ , such that $\mu ( { \dot { A } } ) = { \dot { 0 } }$ and $\nu ( \Omega \setminus A ) { \dot { = } } 0$ $\mu$ is said to be absolutely continuous w.r.t $\nu ,$ denoted $\mu \ll \nu , \operatorname { i f } \nu ( A ) = 0 \Longrightarrow \mu ( A ) = 0 . \operatorname { I f } \mu \ll \nu$ and $\nu \ll \mu ,$ , then we say that $\mu \sim \nu ,$ or that they are equivalent.

Definition 5 (Radon-Nikodym Derivative). If $\mu \ll \nu$ and $\mu , \nu$ are σ−finite, then there exists a measurable function $f : \Omega \stackrel { \cdot } {  } [ 0 , \infty ]$ such that for all $A \in { \mathcal { F } } ;$

$$
\mu ( A ) = \int _ { A } f d \nu ,
$$

where $f$ is called the Radon-Nikodym derivative of $\mu$ w.r.t $\nu ,$ denoted $\frac { d \mu } { d \nu }$ . When the measure space is $( \mathbb { R } ^ { d } , B ( \mathbb { R } ^ { d } ) )$ and $\nu$ is the Lebesgue measure, $f$ is also called the probability density function of distribution $\mu .$

Though there is no infinite-dimensional Lebesgue measure, and hence no universal way to define density, the most feasible and well-understood way to have a Lebesgue-like measure is to define Gaussian measures:

Definition 6 (Gaussian Measure). Let $a \in \mathcal H$ and $Q \in L _ { 1 } ^ { + } ( \mathcal { H } )$ , the space of positive, symmetric, and trace class linear operators, then the Gaussian measure $\mu : = N _ { a , Q }$ on $( { \mathcal { H } } , { \bar { B } } ( { \mathcal { H } } ) )$ ) with mean a and covariance operator $Q$ is defined by the Fourier transform:

$$
\widehat { N _ { a , Q } } ( h ) = \exp \{ i \langle a , h \rangle - \frac { 1 } { 2 } \langle Q h , h \rangle \}
$$

It can be shown that for every choice of $a , Q$ , there exists a unique Gaussian measure. In Da Prato [8], Chapter 1, the interesting argument for constructing this measure is by the theory of infinite product measure and the isomorphism between the sequence space $l ^ { 2 }$ and the Hilbert space $\mathcal { H }$ , where the projection map defined using the indices of the spectral decomposition is mapped with Gaussian measures in R (also known as 1D Gaussian distribution).

Still, for one Gaussian measure to be absolutely continuous with respect to another, strong conditions have to be imposed on their means and covariance operators: the Feldman–Hájek theorem and, for a shifted mean, the Cameron–Martin formula.

Definition 7 (Cameron–Martin Space). Let H be a Hilbert space and $\mu = N _ { 0 , Q }$ a centered Gaussian measure, then the Cameron–Martin space induced by $N _ { 0 , Q }$ is:

$$
H _ { \mu } : = \mathbb { R } \mathrm { a n } ( Q ^ { 1 / 2 } ) ,
$$

with inner product $\langle h _ { 1 } , h _ { 2 } \rangle : = \langle Q ^ { - 1 / 2 } h _ { 1 } , Q ^ { - 1 / 2 } h _ { 2 } \rangle _ { \mathcal { H } }$

The general dichotomy for Gaussian measures is the Feldman–Hájek theorem:

Theorem 5 (Feldman–Hájek). Let H be a separable Hilbert space and $N _ { a , P } , N _ { b , Q }$ be two Gaussian measures, then either $N _ { a , P } \sim N _ { b , Q }$ or $N _ { a , P } \bot N _ { b , Q } ,$ , and ${ N _ { a , P } } \sim N _ { b , Q } i f f \mathrm { . }$

1. $P ^ { 1 / 2 } ( \mathcal { H } ) = Q ^ { 1 / 2 } ( \mathcal { H } ) = : \mathcal { H } _ { 0 } .$

2. $a - b \in \mathcal { H } _ { 0 } .$

3. $( P ^ { - 1 / 2 } Q ^ { 1 / 2 } ) ( P ^ { - 1 / 2 } Q ^ { 1 / 2 } ) ^ { * } - I$ is Hilbert-Schmidt on $\mathcal { H } _ { 0 }$

Theorem 6 (Cameron–Martin Formula). Let $\mu = N _ { Q } , \nu = N _ { a , Q }$ be two Gaussian measures, then:

1. $I f a \not \in Q ^ { 1 / 2 } ( \mathscr { H } )$ , then $\mu \perp \nu .$

2. $I f a \in Q ^ { 1 / 2 } ( \mathcal { H } )$ , then $\mu \sim \nu .$

3. $I f \mu \sim \nu ,$ then with $h : = Q ^ { - 1 / 2 } a$

$$
\frac { d \nu } { d \mu } ( x ) = \exp \left\{ \widehat { h } ( x ) - \frac { 1 } { 2 } \| h \| _ { \mathcal { H } } ^ { 2 } \right\} ,
$$

where $\widehat { h }$ denotes the Cameron–Martin linearfunctional associated with h.

These are the conditions enforced on the FFM model [25], which relies on the absolute continuity between the marginal and the conditional measure at every time t in the probability flow.

For a general Banach space, which is a complete normed space but need not have an inner product structure or spectral decomposition, the analysis becomes more abstract. For instance, without a coordinate map, the preceding product-measure construction of Gaussian measures no longer applies directly. We refer readers to Ledoux and Talagrand [29] for this broader probability theory, and instead explore Hilbert-space embeddings of probability measures in Section ${ \mathrm { A } } . 3$

## A.2 Reproducing Kernel Hilbert Space

A Reproducing Kernel Hilbert Space is a Hilbert space of functions with a particular structure that makes each point evaluation well-behaved.

Definition 8 (Reproducing Kernel Hilbert Space). Let X be a non-empty set. A Hilbert space, H of functions $f : \bar { \mathcal { X } } \to \mathbb { R }$ is called a Reproducing Kernel Hilbert Space if for every $x \in \mathcal { X }$ , the point evaluation functional $\delta _ { x } : \mathcal { H } \to \mathbb { R }$ defined by $\bar { \delta _ { x } ( f ) } = f ( x )$ is continuous and bounded.

By the Riesz representation theorem, there exists a unique elemen $\kappa _ { x } \in \mathcal { H }$ such that

$$
f ( x ) = \langle f , \kappa _ { x } \rangle _ { \mathcal { H } } \forall f \in \mathcal { H } .
$$

This is the reproducing property that is the defining characteristic of an RKHS.

Since $\kappa _ { x }$ is both a function defined on $\mathcal { X }$ and in H, we have that

$$
\kappa _ { x } ( y ) = \delta _ { y } ( \kappa _ { x } ) = \langle \kappa _ { x } , \kappa _ { y } \rangle _ { \mathcal { H } } ,
$$

where $\kappa _ { y } \in \mathcal { H }$ is the element in H associated with $\delta _ { y }$ . This allows us to define the reproducing kernel, $\kappa ( x , y )$ as:

$$
\kappa ( x , y ) = \langle \kappa _ { x } , \kappa _ { y } \rangle _ { \mathcal { H } } .
$$

From this, it is easy to see that $\kappa : \mathcal { X } \times \mathcal { X }  \mathbb { R }$ is both symmetric and positive definite.

Arguably the core reason to use kernels is that they implicitly define feature maps in high (and potentially infinite) dimensional spaces, which is known colloquially as the “kernel trick”. We next formalize this property using the properties of the RKHS:

Definition 9 (Canonical Feature Maps). Let κ be a kernel with RKHS ${ \mathcal { H } } _ { \kappa }$ . The canonical feature map $\phi : \mathcal { X }  \mathcal { H } _ { \kappa }$ is defined by

$$
\phi ( x ) = \kappa ( \cdot , x ) ,
$$

which satisfies the fundamental identity $\kappa ( x , y ) = \langle \phi ( x ) , \phi ( y ) \rangle$ ⟩, which is known as the “kernel trick”.   
This allows us to compute inner products in the feature space without explicit construction.

In the context of our work, we can understand the cost function in Proposition 3 by making use of the kernel trick.

For some of our kernel choices, it is useful to know when the RKHS is rich enough to approximate arbitrary continuous functions and when the induced mean embedding distinguishes probability distributions. Kernels with the former property are referred to as “universal”, defined as follows:

Definition 10 (Universal Kernel). Let X be a compact metric space. A continuous kernel on X is called universal if its associated RKHS, $\mathcal { H } _ { \kappa }$ is dense in C(X) with respect to the supremum norm, $| | \cdot | | _ { \infty }$ . That is to say that for any $f \in C ( \mathcal { X } )$ and for any $\epsilon > 0$ , there exists $g \in \mathcal { H } _ { \kappa }$ such that

$$
\operatorname* { s u p } _ { x \in \mathcal { X } } | f ( x ) - g ( x ) | < \epsilon .
$$

Universality ensures that a kernel can approximate continuous functions, while characteristicness ensures that its mean embedding can distinguish probability distributions. A kernel with the latter property is called “characteristic”:

Definition 11 (Characteristic Kernel). Let X be a compact metric space. A kernel κ on X is characteristic if the mean embedding map

$$
\mu \mapsto \int _ { \mathcal X } \kappa ( \cdot , x ) d \mu ( x )
$$

is injective on the space of probability measures $\mathcal { P } ( \mathcal { X } )$ . Equivalently, κ is characteristic if

$$
\int \int \kappa ( x , y ) d \mu ( x ) d \mu ( y ) - 2 \int \int \kappa ( x , y ) d \mu ( x ) d \nu ( y ) + \int \int \kappa ( x , y ) d \nu ( x ) d \nu ( y ) = 0
$$

implies that $\mu = \nu .$

Universal kernels on compact spaces are characteristic, but the converse does not hold in general. These properties are useful for interpreting kernel discrepancies: characteristicness makes the kernel mean embedding injective, while universality gives a stronger approximation property on compact sets. The main theorems in Section 3.2 only require the specific assumptions stated there, namely bounded positive-definite kernels, compactness for the error decomposition, and Sobolev regularity for discretization invariance.

The radial basis function kernel is used in our Sobolev-space experiments. On a Sobolev space $H ^ { k } ( \Omega )$ it is defined as

$$
\kappa _ { \mathrm { R B F } } ^ { ( k ) } ( f , g ) = \exp \left\{ - \frac { \| f - g \| _ { H ^ { k } } ^ { 2 } } { 2 \sigma ^ { 2 } } \right\} ,
$$

where $\sigma > 0$ is the bandwidth parameter. This Gaussian kernel is symmetric, positive definite, and characteristic on separable Hilbert spaces [54]. It is also bounded, $\kappa _ { \mathrm { R B F } } ^ { ( k ) } ( f , f ) = 1$ , satisfying the boundedness condition used in Theorems 1 and 2.

## A.3 Kernel Embedding of Probability Measure

We next turn our attention to the concept of embedding probability measures into a RKHS, following the framework developed by Sriperumbudur et al. [46]. This is central to our approach because it gives a tractable way to define kernel-induced transport costs on function spaces. As previously discussed, infinite-dimensional spaces lack a Lebesgue-like reference measure, making density-based approaches unavailable. The key insight is that we can embed probability measures as elements into a RKHS, where we compute distances between distributions without requiring explicit density estimation.

For this embedding to be well defined, we require an integrability condition on the kernel. The following proposition shows that bounded kernels ensure the embedding is valid for all probability measures [46]:

Proposition 1. Let κ be a measurable kernel on X. Then, $\begin{array} { r } { \int _ { \mathcal { X } } \sqrt { \kappa ( x , x ) } d P ( x ) < \infty \forall P \in \mathcal { P } ( \mathcal { X } ) } \end{array}$ ifand only $i f \kappa$ is bounded.

With the mean embedding in hand, we can define a natural notion of distance between probability measures. This construct relies on the concept of characteristic kernel (Definition 11). If the kernel is characteristic, then the metrics $\gamma _ { \kappa }$ can be defined such that:

$$
\gamma _ { \kappa } ( \mathbb { P } , \mathbb { Q } ) = 0 \Longleftrightarrow \mathbb { P } = \mathbb { Q } , \mathbb { P } , \mathbb { Q } \in \mathcal { P }
$$

The bounded condition and the characteristic condition gives the following result, which is Theorem 1 in Sriperumbudur et al. [46]:

Theorem 7. Let $\begin{array} { r } { \mathcal { P } _ { \kappa } : = \{ \mathbb { P } \in \mathcal { P } : \int _ { M } \sqrt { \kappa ( x , x ) } d \mathbb { P } ( x ) < \infty \} } \end{array}$ , where κ is measurable on M, then for any $\mathbb { P } , \mathbb { Q } \in \mathcal { P } _ { \kappa } .$

$$
\gamma _ { \kappa } ( \mathbb { P } , \mathbb { Q } ) = \left| \left| \int _ { M } \kappa ( \cdot , x ) d \mathbb { P } ( x ) - \int _ { M } \kappa ( \cdot , x ) d \mathbb { Q } ( x ) \right| \right| _ { \mathcal { H } } : = | | \mathbb { P } \kappa - \mathbb { Q } \kappa | | _ { \mathcal { H } }
$$

Below we show that the bounded RBF kernel defined on a Sobolev space $H ^ { k }$ is characteristic, to provide additional support for the kernel choices in the main text. First we state a lemma proved in Sriperumbudur et al. [46]:

Lemma 1. Let κ be an integrally strictly positive definite kernel on a topological space M. Then κ is characteristic to $\mathcal { P }$

Definition 12. Let $M ( H )$ be the finite signed Borel measures on H. A bounded measurable kernel κ on H is integrally strictly positive definite (ISPD) if for every $\mu \in M ( H ) \setminus \{ 0 \}$ ,

$$
\int \int _ { H \times H } \kappa ( x , y ) d \mu ( x ) d \mu ( y ) > 0 ,
$$

or equivalently, if $\begin{array} { r } { m _ { \kappa } ( \mu ) : = \int \kappa ( \cdot , x ) d \mu ( x ) } \end{array}$ is the kernel mean element (KME), then:

$$
\int \int \kappa ( x , y ) d \mu ( x ) d \mu ( y ) = | | m _ { \kappa } ( \mu ) | | _ { \mathcal { H } _ { \kappa } } ^ { 2 } .
$$

One can interpret ISPD as the statement that a zero KME implies a zero measure: $m _ { \kappa } ( \mu ) = 0 \Longrightarrow$ $\mu = 0$

We then reference Theorem 3.1 from Ziegel et al. [54], which shows that RBF / Gaussian kernel is ISPD on separable Hilbert space:

Theorem 8. Let H be a separable Hilbert space, then the Gaussian kernel on H is ISPD with respect to $M ( H )$

This theorem and Lemma 1 therefore imply that the RBF kernel is characteristic on our Sobolev spaces of interest.

## A.4 Path Space and Signature Kernel

Denote by $\mathcal { C } _ { p }$ the space of continuous paths $x : [ a , b ] \to V$ of finite $p \mathrm { - }$ variation (see Definition 1.1.4 in Cass and Salvi [5]), with $p \in [ 1 , 2 )$ . V is a finite-dimensional Banach space. For time series datasets, we usually have $x : [ 0 , T ] \to \mathbb { R } ^ { d }$ . For discrete observations, one may work with a piecewise-linear interpolation, which is again of finite p-variation. Let $\tau$ be the tensor algebra (written $T ( ( \mathbb { R } ^ { d } ) )$ in Section 2.3, where $V = \mathbb { R } ^ { d } )$

$$
{ \mathcal { T } } : = \prod _ { m = 0 } ^ { \infty } V ^ { \otimes m } .
$$

Definition 13 (The signature transform). The signature transform $S : { \mathcal { C } } _ { p } \to \tau$ maps a path x to the sequence of tensors

$$
S ( x ) = \bigl ( 1 , S ^ { ( 1 ) } ( x ) , S ^ { ( 2 ) } ( x ) , \ldots \bigr ) ,
$$

where the m-th tensor is given by the iterated integral

$$
S ^ { ( m ) } ( x ) : = \int _ { a \leq t _ { 1 } < \cdots < t _ { m } \leq b } d x _ { t _ { 1 } } \otimes \cdots \times \otimes d x _ { t _ { m } } \in V ^ { \otimes m } , \qquad m \in \mathbb { N } .
$$

Here ⊗ denotes the standard tensor product.

One can view $S ( x )$ as an infinite-dimensional feature representation of the path x, with higher signature levels $S ^ { ( m ) } ( x )$ (the level-m term of $S ( x )$ ; we use this notation throughout) encoding increasingly high-order interactions between the d channels.

Definition 14 (Time augmentation). The signature is invariant under continuous and non-decreasing time reparametrizations [5, Lemma 1.2.1]. In many machine learning settings, this invariance is not desirable, as we want to capture time series in a way that also depends on their time parametrisation. Following prior work $[ 2 , 2 6 ]$ , we therefore use the standard time augmentation trick. For a path $x : [ a , b ] { \overset { } { \to } } V$ , we define the augmented path as

$$
\bar { x } : [ a , b ] \to \mathbb { R } \times V , \qquad \bar { x } ( t ) : = ( t , x ( t ) ) .
$$

Throughout, we apply the signature transform to x¯ rather than x. For notational simplicity, we suppress the bar in the definitions below. Note the signature is also invariant under constant translations of x (see Section 1.4 in Cass and Salvi [5]). Throughout, we either fix $x ( a ) = 0$ or treat paths modulo translation.

Remark 1 (Tree-like equivalence without time augmentation). Without the additional time coordinate, the signature is injective only up to tree-like equivalence [5, Section 1.4]. Accordingly, statements such as universality and characteristicness of the signature kernel on a set K should be understood as holding on the corresponding quotient space of equivalence classes when time augmentation is not used.

When working directly with signatures, one typically truncates $S ( x )$ at level $L ,$ defining the truncated signature $S _ { < L } ( x ) = ( 1 , S ^ { ( 1 ) } ( x ) , \ldots , S ^ { ( L ) } ( x ) )$ ). However, if one is interested in comparing paths $x , y \in { \mathcal { C } } _ { p }$ and defining the corresponding kernel-induced discrepancies between distributions, the signature kernel provides a way to do so without truncation by working with inner products on the full tensor algebra. Concretely, equip V with an inner product $\langle \cdot , \cdot \rangle _ { V }$ . This induces canonical inner products $\langle \cdot , \cdot \rangle _ { m }$ on each tensor space $V ^ { \otimes m }$ , and hence an inner product $\langle \cdot , \cdot \rangle _ { \mathcal { T } }$ on $\tau$ by linearity across levels.

Definition 15 (Signature kernel). For two paths $x , y \in { \mathcal { C } } _ { p }$ , the signature kernel $\kappa _ { \mathrm { s i g } } : \mathcal { C } _ { p } \times \mathcal { C } _ { p } \to \mathbb { R }$ is defined as the inner product of their signatures,

$$
\kappa _ { \mathrm { s i g } } ( x , y ) : = \langle S ( x ) , S ( y ) \rangle _ { T } .
$$

It is symmetric and positive semidefinite, and thus defines an RKHS on $\mathcal { C } _ { p } .$

Crucially, $\kappa _ { \mathrm { s i g } }$ can be computed without explicitly forming the signature tensors, via the signature kernel trick introduced in Salvi et al. [42, Theorem 2.5]. In particular, one can evaluate

$$
\kappa _ { \mathrm { s i g } } ( x , y ) = K ( b , b ) ,
$$

where $K \ : \ [ a , b ] ^ { 2 } \  \ \mathbb { R }$ is the signature kernel between two path prefixes, $K ( s , t ) \quad : = $ $\langle S ( x _ { [ a , s ] } ) , S ( \dot { y _ { [ a , t ] } } \dot { ) } \rangle$ , which satisfies the integral equation

$$
K ( s , t ) = 1 + \int _ { a } ^ { s } \int _ { a } ^ { t } K ( u , v ) \langle d x _ { u } , d y _ { v } \rangle _ { V } .
$$

When x and y are continuously differentiable, this reduces to a linear hyperbolic Goursat-type PDE with boundary conditions $K ( { \dot { a } } , \cdot ) = K ( \cdot , a ) = 1$ , for which standard finite-difference solvers apply. See Salvi et al. [42, Section 3.1] for one possible finite-difference scheme. We use the PYSIGLIB library from Shmelev and Salvi [44] for efficient signature kernel computations.

Proposition 2 (Universality of the signature kernel [5]). Let C be a path space equipped with a topology such that the signature transform $S : { \mathcal { C } }  { \mathcal { T } }$ is continuous into a tensor Hilbert space $\tau$ for which the signature kernel

$$
\kappa _ { \mathrm { s i g } } ( x , y ) : = \langle S ( x ) , S ( y ) \rangle _ { T }
$$

is well-defined. Then on compact $K \subset { \mathcal { C } } ,$ , the restriction $\kappa _ { \mathrm { s i g } } | _ { K \times K }$ is universal in the sense of Definition 10, and hence also characteristic to ${ \mathcal { P } } ( K )$

A precise statement of universality and its proof are given in Cass and Salvi [5, Theorem 2.2.3]. Characteristicness on compact K follows from the relation between universality and characteristicness established in Cass and Salvi [5, Theorem 2.2.7].

For a more detailed overview of signatures and signature kernels, we refer the reader to Cass and Salvi [5]. The compact-set universality and characteristicness statements used here can be found specifically in Chapter 2.2.

## A.5 Optimal Transport and Sinkhorn Divergence

This section introduces the static optimal transport problem and its entropy-regularized version, which connects to the theory of statistical divergences.

Let $X , Y$ be metric spaces with Borel σ-algebras $B ( X ) , B ( Y )$ , let $\mu \in { \mathcal { P } } ( X )$ and $\nu \in { \mathcal { P } } ( Y )$ be probability measures, and let $c : X \times Y $ R be a measurable cost function.

Definition 16 (Coupling). Let $\mathrm { p r } _ { X } ( x , y ) = x$ and $\mathrm { p r } _ { Y } ( x , y ) = y$ be the coordinate projections. Then the set of couplings (transport plans) is:

$$
\Pi ( \mu , \nu ) : = { \Bigl \{ } \pi \in { \mathcal { P } } ( X \times Y ) : ( \operatorname { p r } _ { X } ) _ { \# } \pi = \mu , ( \operatorname { p r } _ { Y } ) _ { \# } \pi = \nu { \Bigr \} } .
$$

Definition 17 (Kantorovich Problem). The static optimal transport cost associated with c is

$$
{ \mathsf { O T } } _ { c } ( \mu , \nu ) : = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int _ { X \times Y } c ( x , y ) \pi ( d x , d y ) .
$$

Any minimizer $\pi ^ { \star } \in \Pi ( \mu , \nu )$ (if it exists) is called an optimal transport plan.

Definition 18 (Wasserstein-p metrics). If $X = Y$ is a metric space $( X , d )$ and $c ( x , y ) = d ( x , y ) ^ { p }$ with $p \geq 1$ , define

$$
W _ { p } ( \mu , \nu ) : = \left( \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int _ { X \times X } d ( x , y ) ^ { p } d \pi ( x , y ) \right) ^ { 1 / p } ,
$$

for $\begin{array} { r } { \mu , \nu \mathrm { i n } \mathcal P _ { p } ( X ) : = \left\{ \rho \in \mathcal P ( X ) : \int d ( x , x _ { 0 } ) ^ { p } d \rho ( x ) < \infty \right\} } \end{array}$

In the main text, $\mathcal { W } ( \mu , \nu )$ denotes ${ \mathsf { O T } } _ { c } ( \mu , \nu )$ with the quadratic cost $c = d ^ { 2 }$ , that is, $W _ { 2 } ^ { 2 } ( \mu , \nu )$ . Next we present the Kantorovich dual form, which clarifies what is required of the ambient space for the infimum in the Kantorovich problem to be attained. First, we define Polish spaces:

Definition 19 (Polish space). A topological space $( X , \tau )$ is called Polish if there exists a metric $d : X \times X \to [ 0 , \infty )$ such that:

1. d generates the topology $\tau , \mathrm { i . e . } \tau = \tau _ { d } ,$ where $\tau _ { d }$ is the metric topology induced by $d ;$

2. $( X , d )$ is complete, i.e. every d-Cauchy sequence in X converges (with respect to $d )$ to a point in X;

3. $( X , d )$ is separable, i.e. there exists a countable dense subset $D \subset X$ such that ${ \overline { { D } } } = X$

Equivalently, a metric space $( X , d )$ is Polish if it is complete and separable.

Theorem 9 (Kantorovich duality on Polish spaces). $L e t \left( X , d _ { X } \right)$ and $( Y , d _ { Y } )$ be Polish spaces, and let $\mu \in { \mathcal { P } } ( X ) , \nu \in { \mathcal { P } } ( Y )$ . Assume $c : X \times Y \to ( - \infty , + \infty ] i s$ lower semicontinuous and bounded from below, and that there exists at least one coupling $\pi _ { 0 } \in \Pi ( \mu , \nu )$ with $\int c d \pi _ { 0 } < \infty$ . Then

$$
\operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int _ { X \times Y } c ( x , y ) d \pi ( x , y ) \ = \ \operatorname* { s u p } \left\{ \int _ { X } \varphi d \mu + \int _ { Y } \psi d \nu : \varphi ( x ) + \psi ( y ) \leq c ( x , y ) \forall ( x , y ) \right\} ,
$$

where the supremum is taken over all Borelfunctions $\varphi : X \to \mathbb { R } \cup \{ - \infty \}$ and $\psi : Y \to \mathbb { R } \cup \{ - \infty \}$ such that $\varphi \in L ^ { 1 } ( \mu )$ and $\psi \in L ^ { 1 } ( \nu )$ . Moreover, the infimum is attained by some optimal plan $\pi ^ { \star } \in \Pi ( \mu , \nu )$

This duality theorem can be found in Villani [49], Chapter 5, and the attainment of the infimum on Polish spaces in Villani [49], Theorem 4.1. For Theorem $^ { 2 , }$ attainment of the infimum holds on both function spaces used in our experiments: a separable Hilbert space with its norm topology is Polish, and so is path space with its product topology [5, Lemma 1.4.13].

The theory of entropy-regularized static OT can be found in Peyré and Cuturi [38], Chapter 4, and its connection with statistical divergences in Chapter 8. We state only the definitions needed for the Hilbert Sinkhorn Divergence of the main text.

The entropy-regularized problem adds to the transport cost a multiple of the Kullback–Leibler (KL) divergence of the plan from the product of its marginals.

Definition 20 (Kullback–Leibler (KL) divergence). Let $( X , { \mathcal { F } } )$ be a measurable space and let $P , Q$ be probability measures on it. The KL divergence of P from Q is defined by

$$
\mathrm { K L } ( P \| Q ) : = \left\{ \int _ { X } \log \left( { \frac { d P } { d Q } } \right) d P , \quad { \mathrm { i f ~ } } P \ll Q , \right.
$$

Equivalently, if $P \ll Q$ then

$$
\mathrm { K L } ( P \| Q ) = \int _ { X } { \frac { d P } { d Q } } \log \left( { \frac { d P } { d Q } } \right) d Q ,
$$

where $\textstyle { \frac { d P } { d Q } }$ denotes the Radon–Nikodym derivative (defined $Q \mathrm { - a . e . ) }$ .

Definition 21 (Entropic OT Functional).

$$
\mathsf { O T } _ { c , \varepsilon } ( \mu , \nu ) : = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \left\{ \int _ { X \times Y } c ( x , y ) d \pi ( x , y ) + \varepsilon \operatorname { K L } ( \pi \parallel \mu \otimes \nu ) \right\}
$$

The Sinkhorn divergence is the debiased version of the entropic OT functional:

Definition 22 (Sinkhorn-Divergence).

$$
\mathsf { S } _ { c , \varepsilon } ( \mu , \nu ) : = \mathsf { O } \mathsf { T } _ { c , \varepsilon } ( \mu , \nu ) - \frac { 1 } { 2 } \mathsf { O } \mathsf { T } _ { c , \varepsilon } ( \mu , \mu ) - \frac { 1 } { 2 } \mathsf { O } \mathsf { T } _ { c , \varepsilon } ( \nu , \nu ) .
$$

This is the entropic OT functional used in the proof of Theorem 2; it is well defined in this generality because our ambient spaces are Polish. The Hilbert Sinkhorn divergence in the main text and in Li et al. [31] is obtained by applying the same self-cost debiasing after restricting the ambient space to an RKHS with its norm-induced topology.

## B Technical Proofs

## B.1 Inherited Results from Li et al. (2021)

For transparency, we separate prior results from our original proofs. Proposition 3 (the equivalence behind Equation (4)) and the two propositions below are all inherited from Li et al. [31]. We restate them here because they are standard ingredients used by our function-space analysis, but we do not claim them as new contributions. Their proofs are therefore referenced directly to the original source.

Proposition 3 (Li et al. [31]). The entropic OT functional underlying Definition 1 can be equivalently formulated as in Equation (4), in the sense that $i f \pi ^ { * }$ minimizes Equation (4), then its pushforward $\mathsf { \Phi } \otimes \phi \mathsf { ) } _ { \# } \pi ^ { * }$ minimizes Equation (3); applying the same self-cost subtraction on both sides yields $S _ { \kappa , \epsilon } ( \mu , \tilde { \nu } ) = S _ { \epsilon } ( \phi _ { \# } \mu , \phi _ { \# } \nu )$

Proof: The proof is given as Theorem 1 in the supplementary material of Li et al. $[ 3 1 ] ^ { 2 }$

Proposition 4. Let $\mu _ { n } , \nu _ { n }$ be the empirical estimators in Equation (5) and $\epsilon , \eta > 0 .$ . Then there exists $N > 0$ such that

$$
\forall n \geq N , \qquad \mathbb { P } \left( | S _ { \epsilon } ( \mu _ { n } , \nu _ { n } ) - S _ { \epsilon } ( \mu , \nu ) | \leq \epsilon \eta \right) = 1 .
$$

Proof: The proof is given as Theorem 2 in the supplementary material of Li et al. [31], same link as before <sup>2</sup>.

Proposition $\bar { \mathsf { s } } . \ L e t \| f \| _ { \mathcal { H } } \leq M$ and consider $B _ { \eta } = \{ f \in \mathcal { H } : | | f | | _ { \mathcal { H } } < \eta \}$ . Let m be the number of basis spanning functions $f$ in $B _ { \eta } .$ . Then for M, η, $\delta , \epsilon > 0 ,$

$$
\mathbb { P } \Big ( | S _ { \epsilon } ( \mu _ { n } , \nu _ { n } ) - S _ { \epsilon } ( \mu , \nu ) | \le \epsilon \eta \Big ) \ge 1 - \delta ,
$$

whenever

$$
n \geq \frac { 1 } { \eta ^ { 2 } } \Bigl ( 2 M ^ { 2 } ( \log ( 2 / \delta ) + m \log ( 2 4 M / \eta ) ) \Bigr ) .
$$

Proof: The proof is given as Proposition 3 in the supplementary material of Li et al. [31], same link as before <sup>2</sup>.

## B.2 Absolute Continuity Behind the CFM/FM Equivalence

The equivalence between the conditional and unconditional flow-matching losses in Section 2.1 rests on an absolute-continuity assumption that is worth making explicit in infinite dimensions. Two FFM conditionals $\mathcal { N } ( t f , \sigma _ { t } ^ { 2 } \dot { C _ { 0 } } )$ and $\dot { N } ( t ^ { \prime } f ^ { \prime } , \sigma _ { t ^ { \prime } } ^ { 2 } C _ { 0 } )$ are typically mutually singular (Feldman–Hájek), so the relevant structure is not conditional-versus-conditional but conditional-versus-mixture at each fixed t: one needs $\mu _ { t } ^ { f } \ll \mu _ { t }$ for ν-a.e. f. A sufficient condition, used implicitly by FFM [25], is that the data measure ν is supported in the Cameron–Martin space $H ( C _ { 0 } ) = \mathrm { R a n } ( C _ { 0 } ^ { 1 / 2 } )$ of the prior, together with $\sigma _ { t }$ bounded below $( \sigma _ { \mathrm { m i n } } > 0$ , which Algorithm 1 enforces). Then, for fixed t, all conditionals $\{ \mathcal { N } ( t f , \sigma _ { t } ^ { 2 } C _ { 0 } ) : f \in \mathrm { s u p p } \nu \}$ are mutually equivalent by the Cameron– Martin formula, hence absolutely continuous with respect to the mixture $\mu _ { t } .$ , and the exchange of marginal and conditional velocities is justified at each t. When the data are rougher than the prior (supp $\nu \not \subset H ( C _ { 0 } ) )$ the equivalence degrades, so the prior roughness should be matched to the data; this is one reason the white-noise base measure is selected on the Navier–Stokes family (Appendix C.5).

How kFFM inherits this structure. At the population level, write $G _ { t } ( f _ { 0 } , f _ { 1 } ) : = t f _ { 1 } + \sigma _ { t } f _ { 0 }$ The kFFM path marginal is $\mu _ { t } ^ { \pi } ~ = ~ ( G _ { t } ) _ { \# } \pi$ , the FFM marginal is $\mu _ { t } ~ = ~ ( G _ { t } ) _ { \# } ( \mu _ { 0 } \otimes \nu )$ , and the kFFM conditional given $f _ { 1 }$ is $\mu _ { t } ^ { f _ { 1 } , \pi } \ = \ G _ { t } ( \cdot , f _ { 1 } ) _ { \# } \pi ( \cdot \ | \ f _ { 1 } )$ , while the FFM conditional is $\mathcal { N } ( t f _ { 1 } , \sigma _ { t } ^ { 2 } C _ { 0 } ) = G _ { t } ( \cdot , f _ { 1 } ) _ { \# } \mu _ { 0 }$ . For the entropic problem with a bounded cost (Theorem 1) the optimal plan has the Gibbs form dπ $/ d ( \mu _ { 0 } \otimes \nu \bar { ) } = \bar { \exp } \{ ( \varphi \oplus \psi - c _ { \kappa } ) / \varepsilon \}$ with bounded potentials, so its density is bounded above and away from zero. Hence $\pi \sim \mu _ { 0 } \otimes \nu$ and $\pi ( \cdot \mid f _ { 1 } ) \sim \mu _ { 0 }$ for ν-a.e. $f _ { 1 } ;$ ; since equivalence of measures is preserved under pushforward, $\mu _ { t } ^ { f _ { 1 } , \pi } \sim \mathcal { N } ( t f _ { 1 } , \sigma _ { t } ^ { 2 } C _ { 0 } )$ and $\mu _ { t } ^ { \pi } \sim \mu _ { t }$ , and every absolute-continuity relation that FFM relies on transfers verbatim to kFFM. Two remarks. First, this is exactly where entropic regularization matters: an unregularized OT plan can be singular with respect to the product measure, and the argument would fail. Second, the minibatch plans used in training are discrete surrogates of this population object (Section 3.1); the statement concerns the coupling they approximate.

## B.3 Proof of Well-posedness and Uniform Boundedness (Theorem 1)

Theorem 1 (Well-posedness and uniform boundedness). Let $( { \mathcal { X } } , \| \cdot \| _ { \mathcal { X } } )$ be a real Banach space and let $d _ { \mathcal { X } } ( f , g ) \ \overset { \cdot } { : = } \ \| f - g \| _ { \mathcal { X } }$ . Let $\mu , \nu \in \mathcal P ( \mathcal X )$ and let $\kappa : \mathcal { X } \times \mathcal { X } $ R be measurable, symmetric, and positive definite with RKHS $\mathcal { H } _ { \kappa } .$ . Assume additionally that κ is bounded: $B _ { \kappa } : =$ $\textstyle \operatorname* { s u p } _ { f \in { \mathcal { X } } } \kappa ( f , f ) < \infty$ . Then the kernel cost satisfies $0 \le c _ { \kappa } ( f , g ) \le 4 B _ { \kappa }$ for all $f , g \in { \mathcal { X } }$ , and for every $\varepsilon > 0$ the Hilbert Sinkhorn divergence $S _ { \kappa , \varepsilon } ( \mu , \nu )$ (Definition 1) isfinitefor al $\mu , \nu \in \mathcal { P } ( \mathcal { X } )$ , with $| S _ { \kappa , \varepsilon } ( \mu , \nu ) | \leq 8 B _ { \kappa }$

Proof: The argument is elementary once one uses the boundedness of the RKHS cost induced by a bounded kernel. We start by writing the Hilbert Sinkhorn divergence in the general form

$$
\begin{array} { r } { S _ { \kappa , \varepsilon } ( \mu , \nu ) : = 0 \mathsf { T } _ { \kappa , \varepsilon } ( \mu , \nu ) - \frac { 1 } { 2 } \mathsf { O } \mathsf { T } _ { \kappa , \varepsilon } ( \mu , \mu ) - \frac { 1 } { 2 } \mathsf { O } \mathsf { T } _ { \kappa , \varepsilon } ( \nu , \nu ) . } \end{array}
$$

By the reproducing property, the canonical feature map $\phi$ of Definition 1 satisfies

$$
\| \phi ( f ) \| _ { \mathcal { H } _ { \kappa } } ^ { 2 } = \langle \kappa ( \cdot , f ) , \kappa ( \cdot , f ) \rangle _ { \mathcal { H } _ { \kappa } } = \kappa ( f , f ) \leq B _ { \kappa } \qquad \mathrm { f o r ~ a l l ~ } f \in \mathcal { X } .
$$

Therefore, for any $f , g \in { \mathcal { X } } ,$

$$
c _ { \kappa } ( f , g ) = \| \phi ( f ) - \phi ( g ) \| _ { \mathcal { H } _ { \kappa } } ^ { 2 } \le 2 \| \phi ( f ) \| _ { \mathcal { H } _ { \kappa } } ^ { 2 } + 2 \| \phi ( g ) \| _ { \mathcal { H } _ { \kappa } } ^ { 2 } \le 4 B _ { \kappa } .
$$

Fix $\varepsilon > 0$ . Since $c _ { \kappa } \geq 0$ and $\mathrm { K L } \geq 0$ , we have $0 \mathsf { T } _ { \kappa . \varepsilon } ( \mu , \nu ) \geq 0$ . To upper bound ${ \mathsf { O T } } _ { \kappa , \varepsilon } ( \mu , \nu )$ , use the feasible coupling $\pi _ { 0 } : = \mu \otimes \nu \in \Pi ( \mu , \nu )$ . Then $\dot { \mathrm { K L } } ( \pi _ { 0 } \lVert \mu \otimes \nu ) = 0$ and hence

$$
\mathsf { O T } _ { \kappa , \varepsilon } ( \mu , \nu ) \le \int _ { \mathcal { X } \times \mathcal { X } } c _ { \kappa } ( f , g ) d ( \mu \otimes \nu ) ( f , g ) \le 4 B _ { \kappa } .
$$

Since $\mu \otimes \nu$ is a probability measure, it follows that $0 \leq 0 \mathsf { T } _ { \kappa , \varepsilon } ( \mu , \nu ) \leq 4 B _ { \kappa } < \infty$ . The same argument applies to ${ \mathsf { O T } } _ { \kappa , \varepsilon } ( \mu , \mu )$ and $0 \mathsf { T } _ { \kappa , \varepsilon } ( \nu , \nu )$ . Since each of the three terms in the definition of $S _ { \kappa , \varepsilon } ( \mu , \nu )$ lies in $[ 0 , 4 B _ { \kappa } ]$ , the expression is finite. Moreover,

$$
\begin{array} { r l } & { | S _ { \kappa , \varepsilon } ( \mu , \nu ) | \leq 0 \mathsf T _ { \kappa , \varepsilon } ( \mu , \nu ) + \frac { 1 } { 2 } \mathsf O \mathsf T _ { \kappa , \varepsilon } ( \mu , \mu ) + \frac { 1 } { 2 } \mathsf O \mathsf T _ { \kappa , \varepsilon } ( \nu , \nu ) } \\ & { \qquad \leq 4 B _ { \kappa } + \frac { 1 } { 2 } ( 4 B _ { \kappa } ) + \frac { 1 } { 2 } ( 4 B _ { \kappa } ) = 8 B _ { \kappa } . \qquad \varTheta } \end{array}
$$

Interpretation. Theorem 1 is the well-posedness step needed in the function-space setting: it guarantees that the kernel cost is uniformly bounded by $4 B _ { \kappa }$ and that the HSD objective is finite for all probability measures on the ambient Banach space whenever the kernel is bounded, with no moment assumption on the measures. The uniform bound is what conditions the entropic solver (Section 3.2). This is precisely the hypothesis that is left implicit in the inherited equivalence proposition from Li et al. [31].

## B.4 Proof of the Error Decomposition (Theorem 2)

Theorem 2 (Error decomposition). Let $( \mathcal { X } , d )$ be a compact metric space and let $\mu , \nu \in { \mathcal { P } } ( { \mathcal { X } } )$ Define the target quadratic cost $c ( x , y ) = d ( x , y ) ^ { 2 }$ . Let $\kappa : \mathcal { X } \times \mathcal { X } $ R be a bounded positivedefinite kernel with RKHS-induced cost $c _ { \kappa } ( x , y ) : = \kappa ( x , x ) + \kappa ( y , y ) - 2 \kappa ( x , y )$ , and let $\Delta _ { \kappa } : =$ $\begin{array} { r } { \operatorname* { s u p } _ { x , y \in \mathcal { X } } | c _ { \kappa } ( x , y ) - c ( x , y ) | } \end{array}$ . Let ${ \mathsf { O T } } _ { \kappa , \varepsilon }$ denote the entropic OTfunctional in Equation (4) with cost $c _ { \kappa } ,$ and let $M : = \mathrm { d i a m } ( \mathcal { X } ) < \infty .$ . Then for every $\varepsilon > 0$ and every $\delta \in ( 0 , M ]$ :

$$
| 0 \mathsf T _ { \kappa , \varepsilon } ( \mu , \nu ) - \mathcal W ( \mu , \nu ) | \leq \Delta _ { \kappa } + 2 \varepsilon \log \mathcal N ( \mathcal X , \delta ; d ) + 8 M \delta ,\tag{6}
$$

where $\mathcal { N } ( \mathcal { X } , \delta ; d )$ is the δ-covering number ofX in the metric d.

Proof. Let $c ( x , y ) = d ( x , y ) ^ { 2 }$ and let $c _ { \kappa }$ be the RKHS-induced cost in Theorem 2. By definition of $\Delta _ { \kappa } ,$ for every coupling $\pi \in \Pi ( \mu , \nu )$

$$
\left| \int c _ { \kappa } d \pi - \int c d \pi \right| \leq \Delta _ { \kappa } .
$$

Therefore the unregularized OT values for the two costs differ by at most $\Delta _ { \kappa }$

For the lower bound, non-negativity of the KL term gives

$$
0 \mathsf { T } _ { \kappa , \varepsilon } ( \mu , \nu ) \ge \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int c _ { \kappa } d \pi \ge \mathcal { W } ( \mu , \nu ) - \Delta _ { \kappa } .
$$

For the upper bound, let $m : = \mathcal { N } ( \mathcal { X } , \delta ; d )$ and choose centers $x _ { 1 } , \ldots , x _ { m }$ whose closed δ-balls cover X. A measurable quantization map $Q : \mathcal { X }  \{ x _ { 1 } , . . . , x _ { m } \}$ with $d ( x , Q ( x ) ) \leq \delta$ is obtained by assigning each point to the first center whose ball contains it, giving a finite Borel partition $\check { C } _ { i } = \check { Q } ^ { - 1 } ( \check { \{ x _ { i } \} } )$ . Since X is compact and $c = d ^ { 2 }$ is continuous, an optimal coupling $\pi ^ { \star }$ for $\mathcal { W } ( \mu , \nu )$ exists.

Set $p _ { i j } : = \pi ^ { \star } ( C _ { i } \times C _ { j } ) , a _ { i } : = \mu ( C _ { i } )$ , and $b _ { j } : = \nu ( C _ { j } )$ . For cells with positive mass, let $\mu _ { i }$ and $\nu _ { j }$ be the conditional restrictions of $\mu$ and ν to $\bar { C } _ { i }$ and $C _ { j } ,$ respectively; zero-mass cells may be assigned arbitrary probability measures on X , since their weights vanish. Define

$$
{ \widehat { \pi } } : = \sum _ { i , j } p _ { i j } \mu _ { i } \otimes \nu _ { j } .
$$

This construction preserves the marginals, so $\widehat { \pi } \in \Pi ( \mu , \nu )$ . On $C _ { i } \times C _ { j }$ with $a _ { i } b _ { j } > 0$ , the density of πb with respect to $\mu \otimes \nu$ is $p _ { i j } / ( a _ { i } b _ { j } )$ , and cells with $a _ { i } b _ { j } = 0$ have $p _ { i j } = 0$ . Hence

$$
\mathrm { K L } ( \widehat { \pi } \| \mu \otimes \nu ) = \mathrm { K L } ( p \| a \otimes b ) \leq 2 \log m ,
$$

where the final inequality follows from $\mathrm { K L } ( p | | a \otimes b ) = H ( a ) + H ( b ) - H ( p ) \leq H ( a ) + H ( b ) \leq$ 2 log m. Moreover, if $x \in C _ { i }$ and $y \in C _ { j }$ , then

$$
| d ( x , y ) ^ { 2 } - d ( x _ { i } , x _ { j } ) ^ { 2 } | \leq \left( d ( x , y ) + d ( x _ { i } , x _ { j } ) \right) | d ( x , y ) - d ( x _ { i } , x _ { j } ) | \leq 4 M \delta .
$$

Applying this once under $\widehat { \pi }$ and once under $\pi ^ { \star }$ gives

$$
\int c d \widehat { \pi } \leq \mathcal { W } ( \mu , \nu ) + 8 M \delta .
$$

Using again $\begin{array} { r } { \int c _ { \kappa } d \widehat { \pi } \leq \int c d \widehat { \pi } + \Delta _ { \kappa } } \end{array}$ , feasibility of πb for the entropic problem yields

$$
{ \mathsf { O T } } _ { \kappa , \varepsilon } ( \mu , \nu ) \leq \mathcal { W } ( \mu , \nu ) + \Delta _ { \kappa } + 8 M \delta + 2 \varepsilon \log m .
$$

Combining the lower and upper bounds and substituting $m = \mathcal { N } ( \mathcal { X } , \delta ; d )$ proves the result.

Euclidean covering-rate special case. If $\mathcal { X } \subset \mathbb { R } ^ { d }$ has bounded radius M, then its covering number satisfies log $\mathcal { N } ( \mathcal { X } , \bar { \delta } ) \leq C \bar { d } \log ( M / \delta )$ for a universal constant $C > 0$ . Substituting this into Theorem 2 gives

$$
| 0 \mathsf T _ { \kappa , \varepsilon } ( \mu , \nu ) - \mathcal W ( \mu , \nu ) | \le \Delta _ { \kappa } + C \bigl ( \varepsilon d \log ( M / \delta ) + M \delta \bigr ) ,
$$

after absorbing constants. When $c _ { \kappa } = c ,$ , this recovers the usual entropic-regularization rate up to constants.

## B.5 Proof of Discretization Invariance (Theorem 3)

Theorem 3 (Discretization invariance). Let $\Omega \subset \mathbb { R } ^ { d }$ be a boundedperiodic domain and let $\alpha > k \ge 0$ Set $\mathcal { X } : = B _ { R } ^ { H ^ { \alpha } } : = \{ f \in H ^ { \alpha } ( \Omega ) : \| f \| _ { H ^ { \alpha } } \leq R \}$ , viewed as a compact subset of $H ^ { k } ( \Omega )$ , and let $\mu , \nu \in \mathcal P ( \mathcal X )$ . For the Fourier basis $\{ e _ { n } \} _ { n \in \mathbb { Z } ^ { d } }$ , let V := span $\{ e _ { n } : | n | ^ { 2 } \leq N ^ { 2 / d } \}$ and let ${ \cal P } _ { N } : H ^ { \alpha } ( \dot { \Omega } ) \stackrel { , } {  } V _ { N }$ denote the orthogonal projection; hence dim $\begin{array} { r } { V _ { N } \asymp N . } \end{array}$ . Write $\mu _ { N } : = ( P _ { N } ) _ { \# } \mu$ and $\nu _ { N } : = ( P _ { N } ) _ { \# } \nu $ . Define the Sobolev-RBF kernel $\kappa _ { \mathrm { R B F } } ^ { ( k ) } ( f , g ) : = \exp ( - \| f - g \| _ { H ^ { k } } ^ { 2 } / 2 \sigma ^ { 2 } )$ on $H ^ { k }$ and write $\mathsf { K O T } _ { \varepsilon } ^ { \left( k \right) } : = \mathsf { O T } _ { \kappa _ { \mathrm { R B F } } ^ { \left( k \right) } , \varepsilon }$ for the corresponding entropic kernel-OT functional from Theorem 2. Then for every $\varepsilon > 0$

$$
\big | \mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu , \nu ) - \mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu _ { N } , \nu _ { N } ) \big | \le \frac { 4 R ^ { 2 } } { \sigma ^ { 2 } } N ^ { - 2 ( \alpha - k ) / d } .
$$

Equivalently, in terms of grid spacing $h = N ^ { - 1 / d }$ , the discretization error decays as $O ( h ^ { 2 ( \alpha - k ) } )$ ).

Notation. Throughout this proof, $\Omega \subset \mathbb { R } ^ { d }$ is the bounded periodic domain, $\{ e _ { n } \} _ { n \in \mathbb { Z } ^ { d } }$ is the Fourier basis on Ω, and the Sobolev norm is

$$
\| f \| _ { H ^ { s } } ^ { 2 } = \sum _ { n \in \mathbb { Z } ^ { d } } ( 1 + | n | ^ { 2 } ) ^ { s } | \hat { f } _ { n } | ^ { 2 } , \qquad s \geq 0 .
$$

$P _ { N }$ denotes the orthogonal projection onto $V _ { N } : = \mathrm { s p a n } \{ e _ { n } : | n | ^ { 2 } \leq N ^ { 2 / d } \}$ , so dim $V _ { N } \asymp N$ and the highest retained mode has frequency $\asymp N ^ { 1 / d }$ . We write $\mathsf { K O T } _ { \varepsilon } ^ { \left( k \right) } : = \mathsf { O T } _ { \kappa _ { \mathrm { R B F } } ^ { \left( k \right) } , \varepsilon }$ and

$$
c _ { \mathrm { R B F } } ^ { ( k ) } ( f , g ) : = \| \phi ( f ) - \phi ( g ) \| _ { \mathcal { H } _ { \kappa _ { \mathrm { R B F } } } ^ { ( k ) } } ^ { 2 } = 2 \big ( 1 - \kappa _ { \mathrm { R B F } } ^ { ( k ) } ( f , g ) \big )
$$

for the RKHS-induced cost associated with the RBF kernel on $H ^ { k }$ . Since $B _ { R } ^ { H ^ { \alpha } }$ is compact in $H ^ { k }$ by the compact Sobolev embedding on bounded periodic domains [12] and $c _ { \mathrm { R B F } } ^ { ( k ) }$ is bounded and continuous, the entropic kernel-OT functionals below attain minimizers.

Step 1: Sobolev approximation on the truncation tail. For any $f \in H ^ { \alpha } ( \Omega )$ with $\| f \| _ { H ^ { \alpha } } \leq R$ and any $0 \leq k < \alpha$

$$
\begin{array} { r l } {  { \| f - P _ { N } f \| _ { H ^ { k } } ^ { 2 } = \sum _ { | n | ^ { 2 } > N ^ { 2 / d } } ( 1 + | n | ^ { 2 } ) ^ { k } | \hat { f } _ { n } | ^ { 2 } } } \\ & { = \sum _ { | n | ^ { 2 } > N ^ { 2 / d } } ( 1 + | n | ^ { 2 } ) ^ { - ( \alpha - k ) } ( 1 + | n | ^ { 2 } ) ^ { \alpha } | \hat { f } _ { n } | ^ { 2 } } \\ & { \le ( 1 + N ^ { 2 / d } ) ^ { - ( \alpha - k ) } \| f \| _ { H ^ { \alpha } } ^ { 2 } \le R ^ { 2 } N ^ { - 2 ( \alpha - k ) / d } . } \end{array}
$$

Hence

$$
\| f - P _ { N } f \| _ { H ^ { k } } \ \leq \ R N ^ { - ( \alpha - k ) / d } \qquad { \mathrm { f o r ~ a l l ~ } } f \in B _ { R } ^ { H ^ { \alpha } } .\tag{7}
$$

Step 2: Stability of the squared $H ^ { k }$ distance under simultaneous projection. For $f , g \in B _ { R } ^ { H ^ { \alpha } }$ since $P _ { N }$ is an orthogonal projection on $H ^ { k }$

$$
\begin{array} { r l } & { \| f - g \| _ { H ^ { k } } ^ { 2 } - \| P _ { N } ( f - g ) \| _ { H ^ { k } } ^ { 2 } = \| ( I - P _ { N } ) ( f - g ) \| _ { H ^ { k } } ^ { 2 } } \\ & { \qquad \leq \| f - g \| _ { H ^ { \alpha } } ^ { 2 } \cdot N ^ { - 2 ( \alpha - k ) / d } } \\ & { \qquad \leq 4 R ^ { 2 } N ^ { - 2 ( \alpha - k ) / d } , } \end{array}
$$

where we used $\| f - g \| _ { H ^ { \alpha } } \leq 2 R$ and the Sobolev tail estimate from Step 1 applied to $f - g$

Step 3: Cost stability for the RBF kernel. Let $h ( s ) : = e ^ { - s / ( 2 \sigma ^ { 2 } ) }$ for $s \geq 0 ,$ , so $\kappa _ { \mathrm { R B F } } ^ { ( k ) } ( f , g ) =$ $h ( \| f - g \| _ { H ^ { k } } ^ { 2 } )$ and $\begin{array} { r } { | h ^ { \prime } ( s ) | \leq \frac { 1 } { 2 \sigma ^ { 2 } } } \end{array}$ . Then

$$
\begin{array} { l } { \displaystyle \big | c _ { \mathrm { R B F } } ^ { ( k ) } ( f , g ) - c _ { \mathrm { R B F } } ^ { ( k ) } ( P _ { N } f , P _ { N } g ) \big | = 2 \big | h ( \| f - g \| _ { H ^ { k } } ^ { 2 } ) - h ( \| P _ { N } ( f - g ) \| _ { H ^ { k } } ^ { 2 } ) \big | } \\ { \displaystyle \qquad \leq \frac { 1 } { \sigma ^ { 2 } } \big | \| f - g \| _ { H ^ { k } } ^ { 2 } - \| P _ { N } ( f - g ) \| _ { H ^ { k } } ^ { 2 } \big | \ \leq \ \frac { 4 R ^ { 2 } } { \sigma ^ { 2 } } N ^ { - 2 ( \alpha - k ) / d } . } \end{array}
$$

The bound is uniform over $( f , g ) \in B _ { R } ^ { H ^ { \alpha } } \times B _ { R } ^ { H ^ { \alpha } }$

Step 4: Upper bound. Let $\pi ^ { \star } \in \Pi ( \mu , \nu )$ be a feasible coupling for the continuum problem and define $\widetilde { \pi } _ { N } \overset {  } { : = } ( P _ { N } \otimes P _ { N } ) _ { \# } \pi ^ { \star } \in \Pi ( \mu _ { N } , \nu _ { N } )$ . By Step 3,

$$
\int _ { { V _ { N } \times V _ { N } } } c _ { \mathrm { \tiny { R B F } } } ^ { ( k ) } d \widetilde { \pi } _ { N } \ = \ \int _ { { \mathcal { X } \times \mathcal { X } } } c _ { \mathrm { \tiny { R B F } } } ^ { ( k ) } ( P _ { N } f , P _ { N } g ) d \pi ^ { \star } ( f , g ) \ \leq \ \int c _ { \mathrm { \tiny { R B F } } } ^ { ( k ) } d \pi ^ { \star } + \frac { 4 R ^ { 2 } } { \sigma ^ { 2 } } N ^ { - 2 ( \alpha - k ) / d } .
$$

By the data-processing inequality for KL divergence applied to the measurable map $P _ { N } \otimes P _ { N }$

$$
\mathrm { K L } ( \widetilde { \pi } _ { N } \| \mu _ { N } \otimes \nu _ { N } ) \ \leq \ \mathrm { K L } ( \pi ^ { \star } \| \mu \otimes \nu ) .
$$

Taking $\pi ^ { \star }$ to be an optimizer of $\mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu , \nu )$ and using $\widetilde { \pi } _ { N }$ as a feasible point for $\mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu _ { N } , \nu _ { N } )$ gives

$$
\mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu _ { N } , \nu _ { N } ) \le \mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu , \nu ) + \frac { 4 R ^ { 2 } } { \sigma ^ { 2 } } N ^ { - 2 ( \alpha - k ) / d } .\tag{8}
$$

Step 5: Lower bound via gluing. Let $\pi _ { N } ^ { \star } \in \Pi ( \mu _ { N } , \nu _ { N } )$ be an optimizer for $\mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu _ { N } , \nu _ { N } )$ Disintegrate $\mu$ along $P _ { N } \mathrm { : }$ since compact metric spaces are standard Borel, the standard regular conditional probability theorem applies [10]. Thus there exist Borel kernels $\mu ( \cdot | u ) , u \in V _ { N }$ , such that

$$
\mu ( A ) = \int _ { V _ { N } } \mu ( A \mid u ) d \mu _ { N } ( u ) , \qquad A \subset \mathcal { X } \ \mathrm { B o r e l } ,
$$

with $\mu ( \cdot | u )$ supported on the fibre $P _ { N } ^ { - 1 } ( u )$ for $\mu _ { N } { \bf - } \mathbf { a . e . } \ u .$ , and similarly for $\nu ( \cdot | v )$ . Define the lifted coupling

$$
{ \widehat { \pi } } ( A \times B ) : = \int _ { V _ { N } \times V _ { N } } \mu ( A \mid u ) \nu ( B \mid v ) d \pi _ { N } ^ { \star } ( u , v ) , \qquad A , B \subset { \mathcal { X } } { \mathrm { ~ B o r e l } } .
$$

Direct verification gives ${ \widehat { \pi } } \in \Pi ( \mu , \nu )$ and $( P _ { N } \otimes P _ { N } ) _ { \# } \widehat { \pi } = \pi _ { N } ^ { \star }$ . Since $\mu _ { N } \otimes \nu _ { N }$ is feasible with finite KL, the optimizer $\pi _ { N } ^ { \star }$ has finite KL and hence π<sup>⋆</sup> ≪ µ<sub>N</sub> ⊗ ν<sub>N</sub>. On each fibre product, the Radon–Nikodym derivative of πb with respect to $\mu \otimes \nu$ is exactly the derivative of $\pi _ { N } ^ { \star }$ with respect to $\mu _ { N } \otimes \nu _ { N }$ evaluated at $( u , v ) = ( P _ { N } f , \bar { P _ { N } } g )$ . Therefore, by the chain rule for KL divergence,

$$
\mathrm { K L } ( \widehat { \pi } \| \mu \otimes \nu ) = \mathrm { K L } ( \pi _ { N } ^ { \star } \| \mu _ { N } \otimes \nu _ { N } ) .
$$

For the cost integral, Step 3 applied in reverse gives

$$
\int c _ { \mathrm { R B F } } ^ { ( k ) } d \widehat { \pi } \leq \int _ { V _ { N } \times V _ { N } } c _ { \mathrm { R B F } } ^ { ( k ) } d \pi _ { N } ^ { \star } + \frac { 4 R ^ { 2 } } { \sigma ^ { 2 } } N ^ { - 2 ( \alpha - k ) / d } ,
$$

using that $( f , g ) \sim \widehat \pi$ projects to $( u , v ) \sim \pi _ { N } ^ { \star }$ and that the cost difference is bounded uniformly. Hence

$$
\mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu , \nu ) \le \mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu _ { N } , \nu _ { N } ) + \frac { 4 R ^ { 2 } } { \sigma ^ { 2 } } N ^ { - 2 ( \alpha - k ) / d } .\tag{9}
$$

Step 6: Conclusion. Combining (8) and (9) yields

$$
\big | \mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu , \nu ) - \mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu _ { N } , \nu _ { N } ) \big | \le \frac { 4 R ^ { 2 } } { \sigma ^ { 2 } } N ^ { - 2 ( \alpha - k ) / d } . \qquad \big \Pi
$$

## B.6 Composed End-to-End Bound

Corollary 1 (Composed end-to-end bound). Let $\mathcal { X } = B _ { R } ^ { H ^ { \alpha } }$ be as in Theorem $^ { 3 , }$ equipped with $d _ { k } ( f , g ) : = \| f - g \| _ { H ^ { k } }$ , and let

$$
\mathcal W _ { k } ( \mu , \nu ) : = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int _ { \mathcal X \times \mathcal X } \| f - g \| _ { H ^ { k } } ^ { 2 } d \pi ( f , g )
$$

denote quadratic-cost OT in the $H ^ { k }$ metric. Write $M _ { k } \ : = \ \dim ( { \mathcal { X } } ; d _ { k } )$ and let $\begin{array} { r l } { \mathcal { N } _ { k } ( \delta ) } & { { } : = } \end{array}$ $\mathcal { N } ( \mathcal { X } , \delta ; d _ { k } )$ be the corresponding covering number. For the Sobolev-RBF kernel in Theorem $^ { 3 , }$ define the mismatch $\Delta _ { \mathrm { R B F } } : = \Delta _ { \kappa _ { \mathrm { R B F } } ^ { ( k ) } } , i . e .$

$$
\Delta _ { \mathrm { R B F } } : = \operatorname* { s u p } _ { f , g \in \mathcal { X } } \left| 2 \left( 1 - e ^ { - \| f - g \| _ { H ^ { k } } ^ { 2 } / ( 2 \sigma ^ { 2 } ) } \right) - \| f - g \| _ { H ^ { k } } ^ { 2 } \right| \ \leq \ M _ { k } ^ { 2 } + 2 ,
$$

where the bound follows because the RBF-induced cost lies in $[ 0 , 2 ]$ and $\| f - g \| _ { H ^ { k } } ^ { 2 } \in [ 0 , M _ { k } ^ { 2 } ]$ on X. Then for every $\varepsilon > 0 , \delta \in ( 0 , M _ { k } ]$ and every projection rank $\bar { N } \in \mathbf { \bar { N } } ,$

$$
\begin{array} { r } { \big | \mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu _ { N } , \nu _ { N } ) - \mathcal { W } _ { k } ( \mu , \nu ) \big | \le \underbrace { \Delta _ { \mathrm { R B F } } } _ { k e r n e l - c o s t m i s m a t c h } + \underbrace { 2 \varepsilon \log \mathcal { N } _ { k } ( \delta ) } _ { e n t r o p i c r e g u l a r i z a t i o n } } \\ { + \underbrace { 8 M _ { k } \delta } _ { c o v e r i n g } + \underbrace { \frac { 4 R ^ { 2 } } { \sigma ^ { 2 } } N ^ { - 2 ( \alpha - k ) / d } } _ { d i s c r e t i z a t i o n } . } \end{array}
$$

Proof. By the triangle inequality, $\begin{array} { r } { | \mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu _ { N } , \nu _ { N } ) - \mathcal { W } _ { k } ( \mu , \nu ) | \le | \mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu _ { N } , \nu _ { N } ) - \mathsf { K } _ { \varepsilon } ( \mu _ { N } , \nu _ { N } ) - \mathsf { K } _ { \varepsilon } ( \mu _ { N } , \nu _ { N } ) | } \end{array}$ $\mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu , \nu ) \vert + \vert \mathsf { K O T } _ { \varepsilon } ^ { ( k ) } ( \mu , \nu ) - \mathcal { W } _ { k } ( \mu , \nu ) \vert$ . The first term is bounded by Theorem $_ { 3 ; }$ the second by Theorem 2 applied on the compact metric space $( \mathcal { X } , d _ { k } )$ with the Sobolev-RBF kernel, whose mismatch is $\Delta _ { \mathrm { R B F } } . \bar { \boxminus }$

Remark. Of the four terms, only the discretization term vanishes with projection rank N (rate $N ^ { - 2 ( \alpha - k ) / d }$ , driven by the regularity gap $\alpha - k )$ , the formal counterpart of the empirical resolutioninvariance in Section 4. The corollary relies on compact Sobolev embedding [12], and our PDE benchmarks are used in the corresponding smooth-solution regime; such Sobolev and analytic regularity is standard in the PDE settings considered here, including Navier–Stokes in appropriate regimes [4]. The kernel-cost mismatch $\Delta _ { \mathrm { R B F } }$ , by contrast, depends only on kernel design: zero for the linear kernel, finite for RBF (controlled by data scale and bandwidth σ).

Remark (path space). For data on path space $C _ { p } ( [ 0 , T ] , \mathbb { R } ^ { d } )$ with $p \in [ 1 , 2 )$ , the analogue of spectral truncation is signature truncation at level $L , S _ { \leq L } ( x ) : = ( 1 , S ^ { ( 1 ) } ( x ) , \ldots , S ^ { ( L ) } ( x ) )$ . Known factorial-decay bounds for signatures of finite-p-variation paths [5, Proposition $1 . 4 . 6 ]$ imply that, on compact sets with uniformly bounded p-variation control, the tail $\textstyle \sum _ { \ell > L } \| S ^ { ( \ell ) } ( x ) \| ^ { 2 }$ decays faster than any polynomial in L. By Cauchy–Schwarz, the full and truncated signature kernels are then uniformly close on such compact sets. Consequently, the corresponding entropic kernel-OT objectives differ by at most the uniform signature-kernel cost error, since the KL term is unchanged. This gives the same qualitative discretization-invariance message for path space, but with a factorial tail in the signature truncation level rather than a Sobolev Fourier-tail rate.

## C Additional Experimental Details

This appendix provides implementation details and additional experimental results that complement the main text.

## C.1 Model Parameterization

We implement Functional Flow Matching with the official implementation provided by its authors<sup>3</sup> without modifying any provided hyperparameter of the neural architecture, including $\dot { \sigma } _ { \mathrm { m i n } } = 1 0 ^ { - 4 }$ The baseline models use the following hyperparameters, also following Kerrigan et al. [25]:

1. DDPM: the noise schedule has $\beta _ { 0 } = 1 0 ^ { - 4 } , \beta _ { T } = 0 . 0 2$ , and $T = 1 0 0 0$

2. DDO (NCSN): the time interval is set to $T = 1 0$ , and the noise schedule interpolates geometrically between $\sigma _ { 1 0 } = 1 0 ^ { - 3 }$ and $\sigma _ { 1 } = 1$ on the 1D datasets and between $\sigma _ { 1 0 } \overset { \cdot } { = } 1 0 ^ { - 2 }$ and $\sigma _ { 1 } = 1 0 0$ on the 2D datasets.

3. $\mathrm { G A N O } { \mathrm { : } }$ : the generator is trained every 5 epochs, and the gradient penalty is $\lambda = 0 . 1$ on the 1D and $\lambda = 1 0$ on the 2D datasets.

Training. The baseline models are trained with Adam, with learning rate $1 0 ^ { - 3 }$ on 1D and $5 \times 1 0 ^ { - 4 }$ on 2D data, except that the rate is $1 0 ^ { - 4 }$ for GANO. This configuration also follows Kerrigan et al. [25]; no hyperparameter search was performed on the optimizer.

## C.2 Baseline Description

We describe the baseline generative models used for the empirical comparisons. All operate in function space, and the roster covers the diffusion, flow-matching, adversarial, and score-based paradigms; every baseline uses the same neural-operator backbone and data pipeline as kFFM.

1. DDPM [24]: The DDPM used as a baseline is the functional DDPM model proposed by Kerrigan et al. [24], which generalizes the diffusion model in $\mathbb { R } ^ { d }$ using Gaussian measure theory in Hilbert space. Its formulation is analogous to the base DDPM but performs diffusion directly on Gaussian measures.

2. FFM [25]: Functional Flow Matching is the base model that we extend. It generalizes flow matching in $\mathbb { R } ^ { d }$ to infinite dimensions by formulating a path of Gaussian measures in a Hilbert space, in analogy with Kerrigan et al. [24]. In the terminology of Tong et al. [48], it couples prior and data samples independently, which leaves room for OT-based coupling.

3. CFM-OT(L<sup>2</sup>) [48]: This is the direct finite-dimensional OT baseline used in the main text. It uses the same minibatch Sinkhorn solver and FNO backbone as our method, but applies OT to discretized vectors with the raw $L ^ { 2 }$ cost and the white-noise (i.i.d. Gaussian) base measure, rather than a function-space kernel cost.

4. GANO [40]: The Generative Adversarial Neural Operator (GANO) is a generative adversarial network in function space whose generator maps Gaussian random fields to samples.

5. DDO/NCSN [34]: The Denoising Diffusion Operator (DDO) is the analogue of score-based generative models [45] in infinite dimensions. We follow Kerrigan et al. [25] and use the NCSN noise scale; the loss and noise schedule, including preconditioning, follow the code released with Lim et al. [34].

## C.3 Dataset Description

Of all the datasets we experiment on, AEMET, Gene expression, Economics are from Kerrigan et al. [25], Navier–Stokes, KdV, stochastic Navier–Stokes, stochastic KdV are from Salvi et al. [43], and Heston is from Issa et al. [23]. The descriptions below follow those sources; the architectures and training budgets of our runs are given in Appendix C.4.

AEMET Dataset: This dataset comprises functional observations representing the mean annual profile of average daily temperature (°C) over 1980–2009, collected from 73 weather stations in Spain. Each curve is sampled on a uniform grid of length 365.

Gene Expression: The full dataset contains 10,928 time series sampled at 20 evenly spaced time points, measuring gene-expression amplitudes for four genes. To produce a periodic spiking appearance while preserving the underlying structure, we concatenate the gene-specific sequences in time. Before training, the data are log-transformed and mean-centered. We then focus on 156 high-variability trajectories, selected by requiring the time-averaged standard deviation of each centered series to exceed 0.3.

Economic Dataset: We use the three economic datasets (population, GDP, labor) of Kerrigan et al. [25]; a separate model is trained on each, and the reported Economy metrics are averaged over the component series within each seed (the matched-seed comparisons of Tables 1 and 2 use the population and GDP series). As an example, the population dataset consists of time series tracking population changes for 169 countries worldwide from 1950 to 2018, sampled at 69 discrete time points. For visualization, each country’s series is normalized by its own mean, so the curves show population relative to that country’s average over the 69-year period. Overall, the trajectories display approximately linear growth with a common change point shared across countries.

Heston Dataset: The Heston model [21] is a stochastic volatility model with diffusion-driven, non-smooth sample paths. It models the asset price $S _ { t }$ and its variance $\nu _ { t }$ as (in the standard Heston notation, which is local to this paragraph):

$$
\begin{array} { r l } & { d S _ { t } = \mu _ { t } S _ { t } d t + \sqrt { \nu _ { t } } S _ { t } d W _ { t } ^ { S } } \\ & { d \nu _ { t } = \kappa ( \theta - \nu _ { t } ) d t + \xi \sqrt { \nu _ { t } } d W _ { t } ^ { \nu } , } \end{array}
$$

where $W _ { t } ^ { S } , W _ { t } ^ { \nu }$ are Wiener processes with correlations.

The Heston dataset is generated using the simulation code provided in the official repository of sigker-nsde [23] <sup>4</sup>. The script generates 5000 univariate time series of length 100 on [0, 1]; we model the normalized log-variance paths. A long-horizon variant with 1000 time steps (Heston-Long) is used only in the sensitivity study (the “long volatility paths” of Appendices C.8 and C.16) and in the convergence diagnostic of Figure 6.

KdV and Stochastic KdV Equation: The stochastic Korteweg–de Vries equation is a higher-order SPDE given by:

$$
\partial _ { t } u + \gamma \partial _ { x } ^ { 3 } u = 6 u \partial _ { x } u + \xi , \quad u ( t , 0 ) = u ( t , 1 ) , \quad u ( 0 , x ) = u _ { 0 } ( x ) , \quad ( t , x ) \in [ 0 , T ] \times [ 0 , 1 ] ,
$$

and the base KdV equation simply removes the Q-Wiener process $\xi .$ The KdV dataset is a single trajectory on a grid of 512 spatial points with 201 time steps, and each time slice is one sample; the stochastic KdV dataset consists of 1200 trajectories on a grid of 128 spatial points with 101 time steps each, again treated as snapshots. The simulation code for these equations is taken from the torch-spde repository [43] <sup>5</sup>.

Navier–Stokes and Stochastic Navier–Stokes Equation: The 2D stochastic Navier–Stokes equation for an incompressible flow has the form:

$$
\partial _ { t } w - \nu \Delta w = - u \cdot \nabla w + f + \sigma \xi , \quad w ( 0 , x ) = w _ { 0 } ( x ) , \quad ( t , x ) \in [ 0 , T ] \times [ 0 , 1 ] ^ { 2 } ,
$$

where the deterministic force is a function of space only, and $\xi$ is a Q-Wiener process colored in space and rescaled by $\sigma ~ = ~ 0 . 0 5$ The initial condition is a Gaussian Random Field $w _ { 0 } \sim$ $\mathcal { N } ( \bar { 0 } , 3 ^ { 3 / 2 } ( - \Delta + 4 9 I ) ^ { - 3 } )$ with periodic boundary conditions, and the viscosity is $\nu = 1 0 ^ { - 3 }$ . The base Navier–Stokes equation corresponds to the case where $\sigma = 0$ (no stochastic forcing). We again use the simulation code from torch-spde for these 2D datasets, on a uniform $6 4 \times 6 4$ grid. The turbulent benchmark of Section 4.2 uses the $\nu = 1 0 ^ { - 5 }$ data of Li et al. [32] on the same grid (1200 trajectories; we keep the snapshots at $t \geq 1 0$ and train on 9600 of them), and a $1 2 8 \times 1 2 8$ forced Navier–Stokes dataset is used only for the cost measurements of Appendix C.9.

## C.4 Training Configuration Description

The following tables give the hyperparameters of our runs. In all cases, the base FFM model follows the configuration in the original codebase of Kerrigan et al. [25]. We keep the neural architectures fixed and tune only the OT-related hyperparameters; for the 2D PDE datasets we additionally study the Gaussian-prior length scale in Appendix C.16.

1. All FFM and kFFM runs use the Adam optimizer with learning rate $1 0 ^ { - 3 }$ and a StepLR scheduler (step size $5 0 , \gamma = 0 . 1 )$

2. Table 5 gives the FNO configurations on the 1D sequence datasets and Table 6 those on the 1D PDE datasets. We perform no hyperparameter search for the FNO backbone; only the OT-related hyperparameters are varied, so the configurations are dataset-specific. “Batch Size $\mathrm { ( S i g ) ^ { \dag } }$ is the batch size used with the signature kernel, whose memory grows with path length.

3. The hyperparameters of the FNO on the 2D PDE datasets are in Table 7.

4. Shared training settings are given in Table 8.

Table 5: Training hyperparameters for 1D sequence datasets (FFM with FNO backbone).
<table><tr><td>Dataset</td><td>Epochs</td><td>Batch Size</td><td>Batch Size (Sig)</td><td>FNO Modes</td><td>Width</td><td>MLP Width</td><td>GP l</td><td> $\mathbf { G P } \sigma ^ { 2 }$ </td></tr><tr><td>AEMET</td><td>300</td><td>512</td><td>64</td><td>64</td><td>256</td><td>128</td><td>0.01</td><td>0.1</td></tr><tr><td>Gene Expr.</td><td>300</td><td>512</td><td>128</td><td>16</td><td>256</td><td>128</td><td>0.01</td><td>0.1</td></tr><tr><td>Economy</td><td>300</td><td>512</td><td>128</td><td>16</td><td>128</td><td>64</td><td>0.01</td><td>0.1</td></tr><tr><td>Heston</td><td>300</td><td>512</td><td>128</td><td>32</td><td>256</td><td>128</td><td>0.01</td><td>0.1</td></tr></table>

Table 6: Training hyperparameters for 1D PDE datasets (FFM with FNO backbone).
<table><tr><td>Dataset</td><td>Epochs</td><td></td><td>Batch Size Batch Size (Sig)</td><td>FNO Modes</td><td>Width</td><td>MLP Width</td><td>GP l</td><td> $\mathbf { G P } \sigma ^ { 2 }$ </td></tr><tr><td>KdV</td><td>300</td><td>16</td><td>16</td><td>64</td><td>256</td><td>128</td><td>0.01</td><td>0.1</td></tr><tr><td>Stochastic KdV</td><td>100</td><td>512</td><td>128</td><td>32</td><td>256</td><td>128</td><td>0.01</td><td>0.1</td></tr></table>

## C.5 Configuration Selection Protocol

kFFM as evaluated in this paper has two per-dataset configuration choices: the OT cost entering the entropic coupling (signature kernel, Euclidean–RBF kernel, Sobolev-RBF kernel, or the raw $L ^ { 2 }$ cost) and the Gaussian base measure (a smooth GP prior or white noise). The “kFFM (selected)” column of

Table 7: Training hyperparameters for 2D spatial PDE datasets (FFM with 2D-FNO backbone).
<table><tr><td>Dataset</td><td></td><td></td><td>Epochs Batch Size Batch Size (Sig)</td><td>FNO Modes Hidden Ch. Proj. Ch.</td><td></td><td></td><td>GP l</td><td> $\mathbf { G P } \sigma ^ { 2 }$ </td></tr><tr><td>Navier-Stokes</td><td>300</td><td>512</td><td>512</td><td>16</td><td>32</td><td>64</td><td>0.001</td><td>1.0</td></tr><tr><td>Stoch. NS</td><td>100</td><td>512</td><td>512</td><td>16</td><td>32</td><td>64</td><td>0.001</td><td>1.0</td></tr></table>

Table 8: Common training settings for all datasets.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Optimizer</td><td> $\mathbf { A d a m }$ </td></tr><tr><td>Learning Rate</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>LR Scheduler</td><td>StepLR (step=50, γ=0.1)</td></tr><tr><td> $\operatorname { F F M } \sigma _ { \operatorname* { m i n } }$ </td><td> $1 0 ^ { \dot { - } 4 }$  10 (seeds</td></tr><tr><td>Number of Seeds</td><td> ${ 2 ^ { 0 } , \dots , 2 ^ { 9 } } ;$  20 (seeds  $2 ^ { 0 } , \ldots , 2 ^ { 1 9 } )$  where n = 20 in Table 2; 5 on the turbulent benchmark</td></tr><tr><td>ODE solver tolerances (atol, rtol)</td><td> $1 0 ^ { - 5 }$ </td></tr></table>

Table 1 is determined by a validation-based rule rather than by test-set performance: for each dataset, every candidate configuration is trained identically, the configuration with the best sliced Wasserstein distance — a criterion independent of the kernels being compared — on a held-out validation split is selected, and only the selected configuration is then evaluated on the test set. In practice this rule is nearly deterministic and coincides with a simple prior-knowledge heuristic that practitioners can apply without any search: use the signature kernel with a GP base for path-like or rough sequence data, and a bounded RBF cost (Sobolev-RBF with a GP base on the 1D KdV family, Euclidean-RBF on the Navier–Stokes family) for PDE fields, with the white-noise base selected on the Navier–Stokes family, whose fields are rougher than a smooth GP prior. On our benchmark the validated choice coincides with the best-performing configuration on all datasets except short-horizon Heston, where the top configurations are statistically indistinguishable. Pointwise diagnostics by kernel variant are reported in Appendix $\mathrm { C . 6 } ;$ the cost and base measure of each selected cell are tagged in Table 1, and the Sinkhorn regularization and kernel bandwidth follow the defaults of Appendix C.8, to which the results are insensitive within the ranges reported in the sensitivity study below.

## C.6 Detailed Main-Paper Results

This appendix includes a self-contained summary of the full baseline roster used in the paper, together with structural diagnostics, fuller pointwise comparisons, kernel ablations, and the superresolution visualizations referenced in Section 4. The additional pointwise tables follow the evaluation convention of Kerrigan et al. [25]: discretization-level mean/variance MSE summaries together with autocorrelation or log-spectrum diagnostics, and higher moments where reported. These tables focus on models for which such pointwise statistics are currently available, while Table 1 in the main text summarizes the complete baseline family comparison including CFM-OT(L<sup>2</sup>).

## C.7 Full Statistics Report on Sequence Datasets

For the sequence datasets, in addition to the mean/variance/autocorrelation tables above, we also report kurtosis and skewness in Table 14, since these were also reported in Kerrigan et al. [25]. We keep them in the appendix because they are complementary to the main conclusion rather than central to it.

## C.8 Implementation Details and Good Practices

Algorithm 1 (Section 3.1) summarizes kFFM training; here we collect the implementation choices behind it.

Unless stated otherwise, the Sinkhorn regularization is $\epsilon = 0 . 0 5$ on the median-normalized cost matrix and the RBF bandwidth is $\sigma = 1$ on the same normalized scale (i.e. the median heuristic). The coupling is computed per minibatch with the following choices, which we found to matter for stability and recommend as defaults. (i) Median-normalize the per-batch cost matrix before Sinkhorn; this is the most important stability lever and makes the entropic regularization ϵ scale-free. (ii) Use log-domain Sinkhorn [38]. (iii) Sample endpoint pairs from the plan rather than using the barycentric projection. (iv) Use a bounded (kernel) cost: results are then invariant to ϵ across three orders of magnitude and Sinkhorn converges faster (15–25% faster end-to-end training than with the unbounded ${ \breve { L } } ^ { 2 }$ cost; Table 15); with the raw $L ^ { 2 }$ cost keep $\epsilon \leq 0 . 1$ (we observed a ≈ 10× degradation at $\epsilon \geq 0 . 5$ on long volatility paths). (v) Per-batch re-coupling was stable on all datasets. (vi) Batch size. Minibatch kernel OT induces the expected minibatch plan $\bar { \pi } _ { b }$ (Section 3.1); in a sweep over $b \in \{ 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \}$ on Heston and Stoch. KdV, kFFM tracks FFM across the 8× range, and the only seed-consistent edge favors kFFM at the largest batch (Stoch. KdV, $b = 5 1 2 \colon 3 / 3$ seeds, 0.11 vs. $0 . 5 \dot { 2 } \times 1 0 ^ { - 3 }$ MMD-RBF).

Table 9: Interpretable structural diagnostics complementing the full-baseline MMD comparison in Table 1. The first block reports autocorrelation error on sequence datasets; the second block reports log-spectrum error on PDE datasets. Lower is better. As in Tables 11 and 13, the kFFM column reports the best kernel variant per row (the signature kernel on Gene Expression, Economy, and Heston). All results are averaged over 10 seeds.
<table><tr><td>Dataset</td><td>DDO/NCSN</td><td>GANO</td><td>DDPM</td><td>FFM</td><td>CFM-OT(L2)</td><td>kFFM</td></tr><tr><td>AEMET (Autocorr.)</td><td> $4 . 2 1 { \pm } 0 . 9 4 \times 1 0 ^ { - 3 }$ </td><td> $4 . 4 5 { \pm } 3 . 4 3 \times 1 0 ^ { - 6 }$ </td><td> $2 . 8 4 { \pm } 2 . 9 8 \times 1 0 ^ { - 7 }$ </td><td> $4 . 6 9 { \pm } 1 . 5 3 \times 1 0 ^ { - 7 }$ </td><td> $\mathbf { 1 . 2 3 { \pm } 2 . 7 2 \times 1 0 ^ { - 8 } }$ </td><td> $2 . 6 0 { \pm } 1 . 9 2 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Gene Expr. (Autocorr.)</td><td> $5 . 4 8 { \scriptstyle \pm 0 . 2 4 \times 1 0 ^ { - 2 } }$ </td><td> $1 . 1 6 { \pm } 0 . 6 8 \times 1 0 ^ { - 1 }$ </td><td> $3 . 5 9 { \pm } 3 . 4 6 \times 1 0 ^ { - 4 }$ </td><td> $1 . 5 9 { \pm } 1 . 9 8 \times 1 0 ^ { - 4 }$ </td><td> $2 . 7 8 { \pm } 2 . 3 2 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 4 . 1 8 { \pm } 4 . 0 2 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td>Economy (Autocorr.)</td><td> $2 . 7 3 { \scriptstyle \pm 0 . 9 5 \times 1 0 ^ { - 1 } }$ </td><td> $2 . 2 9 { \pm } 1 . 9 1 \times 1 0 ^ { - 1 }$ </td><td> $2 . 0 5 { \pm } 2 . 8 5 \times 1 0 ^ { - 2 }$ </td><td> $1 . 7 2 { \scriptstyle \pm 1 . 0 2 \times 1 0 ^ { - 4 } }$ </td><td> $9 . 3 5 { \pm } 3 . 7 5 \times 1 0 ^ { - 3 }$ </td><td> $\mathbf { 8 . 9 0 { \overset { - } { \pm } } 8 . 3 9 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td>Heston (Autocorr.)</td><td> $4 . 3 6 { \pm } 0 . 2 5 \times 1 0 ^ { - 2 }$ </td><td> $8 . 3 6 { \pm } 8 . 6 8 \times 1 0 ^ { - 3 }$ </td><td> $3 . 3 5 { \pm } 2 . 8 0 \times 1 0 ^ { - 4 }$ </td><td> $2 . 7 8 { \scriptstyle \pm 2 . 0 5 } \times 1 0 ^ { - 4 }$ </td><td> $1 . 7 3 { \scriptstyle \pm 1 . 1 7 \times 1 0 ^ { - 4 } }$ </td><td> $\mathbf { 3 . 3 1 \pm 3 . 3 8 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td>KdV (Log-spectrum)</td><td> $4 . 5 2 { \pm } 0 . 0 0 \times 1 0 ^ { 1 }$ </td><td> $2 . 0 2 { \pm } 0 . 3 0 \times 1 0 ^ { 1 }$ </td><td> $1 . 7 1 { \pm } 0 . 1 3 \times 1 0 ^ { 1 }$ </td><td> $1 . 7 2 { \scriptstyle \pm 0 . 0 3 \times 1 0 ^ { 1 } }$ </td><td> $2 . 7 3 { \pm } 0 . 0 5 \times 1 0 ^ { 1 }$ </td><td> $\mathbf { 1 . 6 9 { \pm 0 . 0 2 } \times 1 0 ^ { 1 } }$ </td></tr><tr><td>Navier-Stokes (Log-spectrum)</td><td> $9 . 9 3 { \scriptstyle \pm 0 . 0 8 \times 1 0 ^ { - 1 } }$ </td><td> $2 . 2 1 { \pm } 0 . 1 5 \times 1 0 ^ { 0 }$ </td><td> $\mathbf { 2 . 5 8 { \pm 0 . 6 3 \times 1 0 ^ { - 1 } } }$ </td><td> $7 . 9 2 { \pm } 0 . 8 9 \times 1 0 ^ { - 1 }$ </td><td> $3 . 4 4 { \pm } 0 . 3 5 \times 1 0 ^ { - 1 }$ </td><td> $5 . 8 7 { \pm } 0 . 4 6 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>Stoch. KdV (Log-spectrum)</td><td> $1 . 8 4 { \pm } 0 . 0 0 \times 1 0 ^ { 1 }$ </td><td> $1 . 4 5 { \pm } 0 . 0 9 \times 1 0 ^ { 1 }$ </td><td> $\mathbf { 1 . 5 7 { \pm 0 . 2 2 \times 1 0 ^ { 0 } } }$ </td><td> $4 . 2 7 { \pm } 0 . 2 5 \times 1 0 ^ { 0 }$ </td><td> $9 . 2 5 { \pm } 0 . 3 1 \times 1 0 ^ { 0 }$ </td><td> $3 . 9 7 { \scriptstyle \pm 0 . 2 8 \times 1 0 ^ { 0 } }$ </td></tr><tr><td>Stoch.  $\mathsf { N S } \left( \mathsf { L o g - s p e c t r u m } \right)$ </td><td> $1 . 0 6 { \pm } 0 . 2 8 \times 1 0 ^ { 0 }$ </td><td> $1 . 0 3 { \pm } 0 . 9 3 \times 1 0 ^ { 0 }$ </td><td> $8 . 6 0 { \pm } 2 . 3 9 \times 1 0 ^ { - 1 }$ </td><td> $6 . 2 2 { \pm } 1 . 0 5 { \times } 1 0 ^ { - 2 }$ </td><td> $\mathbf { 3 . 1 1 { \pm } 0 . 9 3 \times 1 0 ^ { - 2 } }$ </td><td> $3 . 2 2 { \pm } 0 . 6 3 { \times } 1 0 ^ { - 2 }$ </td></tr></table>

Table 10: Sequence-data pointwise diagnostics following the evaluation convention of Kerrigan et al. [25]: mean and variance of the discretization-level MSE between target and generated samples, together with average autocorrelation error. We compare baseline FFM (Independent) against the OT-coupled variants: Signature = signature kernel, RBF = Euclidean–RBF kernel, Euclidean = raw $L ^ { 2 }$ cost with the GP base (kFFM-Euc). Results are mean±std over 10 seeds. Best is green; second best is orange.
<table><tr><td>Dataset</td><td>Kernel</td><td>Mean</td><td>Variance</td><td>Autocorr.</td></tr><tr><td rowspan="4">AEMET</td><td>Independent</td><td> $7 . 3 8 \pm 6 . 3 7 \times 1 0 ^ { - 2 }$ </td><td> $1 . 2 0 \pm 0 . 4 0 \times 1 0 ^ { - 2 }$ </td><td> $4 . 6 9 \pm 1 . 5 3 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Signature</td><td> $3 . 6 0 \pm 3 . 6 6 \times 1 0 ^ { - 2 }$ </td><td> $7 . 8 4 \pm 5 . 5 9 \times 1 0 ^ { - 3 }$ </td><td> $3 . 8 7 \pm 2 . 4 5 \times { 1 0 ^ { - 7 } }$ </td></tr><tr><td>RBF</td><td> $1 . 5 1 \pm 1 . 6 1 \times 1 0 ^ { - 2 }$ </td><td> $4 . 7 0 \pm 2 . 5 7 \times 1 0 ^ { - 3 }$ </td><td> $2 . 6 0 \pm 1 . 9 2 \times { 1 0 } ^ { - 7 }$ </td></tr><tr><td>Euclidean</td><td> $4 . 1 0 \pm 2 . 6 9 \times 1 0 ^ { - }$ </td><td>-2  $5 . 5 9 \pm 2 . 6 5 \times 1 0 ^ { - }$  -3</td><td> $3 . 4 7 \pm 1 . 9 9 \times 1 0 ^ { - 7 }$ </td></tr><tr><td rowspan="4">Gene Expr.</td><td>Independent</td><td> $1 . 3 9 \pm 0 . 3 0 \times 1 0 ^ { - 3 }$ </td><td> $5 . 1 3 \pm 0 . 7 4 \times 1 0 ^ { - 3 }$ </td><td> $1 . 5 9 \pm 1 . 9 8 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Signature</td><td> $5 . 2 1 \pm 0 . 9 8 \times 1 0 ^ { - 4 }$ </td><td> $2 . 2 3 \pm { 0 . 2 3 } \times 1 0 ^ { - 3 }$ </td><td> $4 . 1 8 \pm 4 . 0 2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>RBF</td><td> $1 . 2 5 \pm 0 . 2 0 \times 1 0 ^ { - 3 }$ </td><td> $5 . 6 7 \pm 0 . 7 4 \times 1 0 ^ { - 3 }$ </td><td> $4 . 5 4 \pm 6 . 5 1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Euclidean</td><td> $1 . 2 5 \pm { 0 . 2 0 } \times 1 0 ^ { - 3 }$ </td><td> $5 . 5 6 \pm 0 . 6 1 \times 1 0 ^ { - 3 }$ </td><td> $4 . 4 5 \pm 4 . 4 7 \times 1 0 ^ { - 5 }$ </td></tr><tr><td rowspan="4">Economy</td><td>Independent</td><td> $4 . 9 0 \pm 3 . 2 6 \times 1 0 ^ { - 5 }$ </td><td> $1 . 0 8 \pm 1 . 1 3 \times 1 0 ^ { - 5 }$ </td><td> $1 . 7 2 \pm 1 . 0 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Signature</td><td> $3 . 3 0 \pm 2 . 2 2 \times 1 0 ^ { - 5 }$ </td><td> $6 . 6 0 \pm 5 . 6 2 \times 1 0 ^ { - 6 }$ </td><td> $8 . 9 0 \pm 8 . 3 9 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>RBF</td><td> $3 . 9 6 \pm 2 . 1 2 \times 1 0 ^ { - 5 }$ </td><td> $6 . 8 2 \pm 3 . 4 4 \times 1 0 ^ { - 6 }$ </td><td> $1 . 3 8 \pm 0 . 8 9 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Euclidean</td><td> $3 . 9 4 \pm 2 . 0 7 \times { 1 0 } ^ { - 5 }$ </td><td> $6 . 7 6 \pm 3 . 4 8 \times 1 0 ^ { - 6 }$ </td><td> $1 . 3 8 \pm 0 . 8 9 \times 1 0 ^ { - 4 }$ </td></tr><tr><td rowspan="4">Heston</td><td>Independent</td><td> $6 . 6 6 \pm 4 . 7 7 \times 1 0 ^ { - 3 }$ </td><td> $1 . 0 7 \pm 0 . 2 5 \times 1 0 ^ { - 1 }$ </td><td> $2 . 7 8 \pm 2 . 0 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Signature</td><td> $3 . 3 3 \pm 2 . 2 5 \times 1 0 ^ { - 3 }$ </td><td> $8 . 1 7 \pm 1 . 1 4 \times 1 0 ^ { - 2 }$ </td><td> $3 . 3 1 \pm 3 . 3 8 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>RBF</td><td> $4 . 4 8 \pm 4 . 1 8 \times 1 0 ^ { - 3 }$ </td><td> $8 . 2 7 \pm 1 . 5 3 \times 1 0 ^ { - 2 }$ </td><td> $1 . 4 1 \pm 1 . 0 3 \times { 1 0 } ^ { - 4 }$ </td></tr><tr><td>Euclidean</td><td> $3 . 3 8 \pm 3 . 8 1 \times 1 0 ^ { - 3 }$ </td><td> $8 . 4 0 \pm 0 . 7 7 \times 1 0 ^ { - 2 }$ </td><td> $9 . 0 8 \pm 7 . 9 9 \times 1 0 ^ { - 5 }$ </td></tr></table>

## C.9 Compute and Memory Cost

Table 15 summarizes the measured cost of the coupling step, profiled on Heston, Stoch. KdV, and Navier–Stokes at $6 4 ^ { 2 }$ and $1 2 8 ^ { 2 }$ for batch sizes $b \in \{ 6 4 , \bar { 1 } 2 8 , 2 5 6 , 5 1 2 \}$ . The end-to-end overhead is $5 \mathrm { - } 7 \%$ at the largest resolution we train $( 1 2 8 ^ { 2 } )$ , and because the ${ \dot { O } } ( b ^ { 2 } )$ coupling cost is independent of model size and resolution, its relative weight shrinks as the backbone grows. The bounded RBF cost also conditions the Sinkhorn iterations (the Gibbs kernel $e ^ { - c _ { \kappa } / \epsilon }$ stays bounded away from 0 and 1 across batches; Theorem 1), which is why it is about 8× faster than unbounded- $. L ^ { 2 }$ Sinkhorn at $b = 5 1 2 .$ . Signature-kernel memory grows with path length, which is why the selection rule assigns RBF or $L ^ { 2 }$ costs to very long sequences.

Table 11: Baseline comparison for time series datasets among models with available pointwise statistics. The full baseline roster, including $\mathrm { C F M - O T } ( L ^ { 2 } )$ , is summarized in Table 1. Following Kerrigan et al. [25], we report mean/variance MSE and autocorrelation error computed on the discretized samples. The kFFM row reports the best of the kernel variants in Table 10 for each metric (an oracle over variants, used only for these diagnostics), whereas Table 1 uses the validation-selected configuration. Results are mean±std over 10 seeds. Best is green; second best is orange.
<table><tr><td>Dataset</td><td>Model</td><td>Mean</td><td>Variance</td><td>Autocorr.</td></tr><tr><td rowspan="5">AEMET</td><td>DDO/NCSN</td><td> $3 . 5 6 \pm 0 . 9 6 \times 1 0 ^ { - 1 }$ </td><td> $1 . 1 7 \pm 0 . 0 3 \times 1 0 ^ { - 1 }$ </td><td> $4 . 2 1 \pm 0 . 9 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>DDPM</td><td> $1 . 9 2 \pm 1 . 2 9 \times 1 0 ^ { - 1 }$ </td><td> $3 . 2 2 \pm 2 . 9 8 \times 1 0 ^ { - 2 }$ </td><td> $2 . 8 4 \pm 2 . 9 8 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>GANO</td><td> $2 . 4 7 \pm 0 . 8 9 \times 1 0 ^ { - 1 }$ </td><td> $2 . 3 5 \pm 0 . 0 1 \times 1 0 ^ { - 1 }$ </td><td> $4 . 4 5 \pm 3 . 4 3 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>FFM</td><td> $7 . 3 8 \pm 6 . 3 7 \times 1 0 ^ { - 2 }$ </td><td> $1 . 2 0 \pm 0 . 4 0 \times 1 0 ^ { - 2 }$ </td><td> $4 . 6 9 \pm 1 . 5 3 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>kFFM</td><td> $1 . 5 1 \pm 1 . 6 1 \times 1 0 ^ { - 2 }$  </td><td> $4 . 7 0 \pm 2 . 5 7 \times 1 0 ^ { - 3 }$  </td><td> $2 . 6 0 \pm 1 . 9 2 \times 1 0 ^ { - 7 }$ </td></tr><tr><td rowspan="5">Gene Expr.</td><td>DDO/NCSN</td><td> $6 . 0 7 \pm 1 . 0 6 \times 1 0 ^ { - 3 }$ </td><td> $9 . 6 9 \pm 0 . 7 0 \times 1 0 ^ { - 3 }$ </td><td> $5 . 4 8 \pm 0 . 2 4 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>DDPM</td><td> $4 . 1 0 \pm 1 . 2 4 \times 1 0 ^ { - 3 }$ </td><td></td><td> $3 . 5 9 \pm 3 . 4 6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>GANO</td><td></td><td> $1 . 0 8 \pm 0 . 0 8 \times 1 0 ^ { - 2 }$ </td><td></td></tr><tr><td>FFM</td><td> $1 . 8 9 \pm 0 . 5 2 \times 1 0 ^ { - 2 }$ </td><td> $3 . 1 4 \pm 0 . 0 0 \times 1 0 ^ { - 2 }$ </td><td> $1 . 1 6 \pm 0 . 6 8 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>kFFM</td><td> $1 . 3 9 \pm 0 . 3 0 \times 1 0 ^ { - 3 }$   $5 . 2 1 \pm 0 . 9 8 \times 1 0 ^ { - 4 }$  </td><td> $5 . 1 3 \pm 0 . 7 4 \times 1 0 ^ { - 3 }$  </td><td> $1 . 5 9 \pm 1 . 9 8 \times 1 0 ^ { - 4 }$   $4 . 1 8 \pm 4 . 0 2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td rowspan="5">Economy</td><td>DDO/NCSN</td><td> $1 . 7 0 \pm 1 . 3 4 \times 1 0 ^ { - 3 }$ </td><td> $2 . 2 3 \pm { 0 . 2 3 } \times { 1 0 ^ { - 3 } }$ </td><td></td></tr><tr><td>DDPM</td><td></td><td> $9 . 9 1 \pm 2 . 0 6 \times 1 0 ^ { - 3 }$ </td><td> $2 . 7 3 \pm 0 . 9 5 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>GANO</td><td> $6 . 0 8 \pm 7 . 9 2 \times 1 0 ^ { - 4 }$ </td><td> $2 . 2 7 \pm 5 . 3 4 \times 1 0 ^ { - 4 }$ </td><td> $2 . 0 5 \pm 2 . 8 5 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>FFM</td><td> $3 . 7 9 \pm 0 . 5 1 \times 1 0 ^ { - 1 }$ </td><td> $7 . 2 8 \pm 4 . 0 8 \times 1 0 ^ { - 4 }$ </td><td> $2 . 2 9 \pm 1 . 9 1 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>kFFM</td><td> $4 . 9 0 \pm 3 . 2 6 \times 1 0 ^ { - 5 }$   $3 . 3 0 \pm 2 . 2 2 \times 1 0 ^ { - }$  5</td><td> $1 . 0 8 \pm 1 . 1 3 \times 1 0 ^ { - 5 }$ </td><td> $1 . 7 2 \pm 1 . 0 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td rowspan="5">Heston</td><td>DDO/NCSN</td><td></td><td> $6 . 6 0 \pm 5 . 6 2 \times 1 0 ^ { - 6 }$ </td><td> $8 . 9 0 \pm 8 . 3 9 \times 1 0 ^ { - 5 }$ </td></tr><tr><td></td><td> $1 . 6 0 \pm 0 . 4 7 \times 1 0 ^ { - 2 }$ </td><td> $7 . 3 2 \pm 0 . 1 5 \times 1 0 ^ { - 1 }$ </td><td> $4 . 3 6 \pm 0 . 2 5 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>DDPM</td><td> $9 . 3 3 \pm 9 . 4 3 \times 1 0 ^ { - 3 }$ </td><td> $1 . 0 8 \pm 0 . 2 5 \times 1 0 ^ { - 1 }$ </td><td> $3 . 3 5 \pm 2 . 8 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>GANO</td><td> $1 . 5 5 \pm 0 . 9 0 \times 1 0 ^ { - 2 }$ </td><td> $4 . 4 1 \pm 1 . 1 5 \times 1 0 ^ { - 1 }$ </td><td> $8 . 3 6 \pm 8 . 6 8 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>FFM kFFM</td><td> $6 . 6 6 \pm 4 . 7 7 \times 1 0 ^ { - 3 }$   $3 . 3 3 \pm 2 . 2 5 \times 1 0 ^ { - 3 }$  </td><td> $1 . 0 7 \pm 0 . 2 5 \times 1 0 ^ { - 1 }$   $8 . 1 7 \pm 1 . 1 4 \times 1 0 ^ { - 2 }$  </td><td> $2 . 7 8 \pm 2 . 0 5 \times 1 0 ^ { - 4 }$   $3 . 3 1 \pm 3 . 3 8 \times 1 0 ^ { - 5 }$ </td></tr></table>

Table 12: PDE pointwise diagnostics following the evaluation convention of Kerrigan et al. [25]: mean and variance of the discretization-level MSE between target and generated samples, together with average log-spectrum error. Here we compare the baseline FFM (Independent) against its OT-coupled variants: RBF = Euclidean–RBF kernel, Euclidean = raw $L ^ { 2 }$ cost with the GP base (kFFM-Euc). Results are mean±std over 10 seeds. Best is green; second best is orange.
<table><tr><td>Dataset</td><td>Kernel</td><td>Mean</td><td>Variance</td><td>Spectrum (log)</td></tr><tr><td rowspan="3">KdV</td><td>Independent</td><td> $5 . 2 0 \pm 3 . 3 7 \times 1 0 ^ { - 4 }$ </td><td> $1 . 3 9 \pm 0 . 4 9 \times 1 0 ^ { - 4 }$ </td><td> $1 . 7 2 \pm 0 . 0 3 \times 1 0 ^ { 1 }$ </td></tr><tr><td>RBF</td><td> $1 . 4 8 \pm 0 . 9 5 \times 1 0 ^ { - 4 }$ </td><td> $5 . 6 8 \pm 2 . 5 8 \times 1 0 ^ { - 5 }$ </td><td> $1 . 6 9 \pm 0 . 0 2 \times 1 0 ^ { 1 }$ </td></tr><tr><td>Euclidean</td><td> $2 . 5 1 \pm 0 . 7 4 \times 1 0 ^ { - 4 }$ </td><td> $6 . 8 8 \pm 3 . 3 0 \times 1 0 ^ { - 5 }$ </td><td> $1 . 6 9 \pm 0 . 0 3 \times 1 0 ^ { 1 }$ </td></tr><tr><td rowspan="3">Navier-Stokes RBF</td><td>Independent</td><td> $7 . 9 6 \pm 7 . 8 2 \times 1 0 ^ { - 2 }$ </td><td> $3 . 1 1 \pm 0 . 8 0 \times 1 0 ^ { - 2 }$ </td><td> $7 . 9 2 \pm 0 . 8 9 \times 1 0 ^ { - 1 }$ </td></tr><tr><td></td><td> $2 . 0 1 \pm 2 . 0 1 \times 1 0 ^ { - 3 }$ </td><td> $7 . 5 1 \pm 3 . 3 3 \times 1 0 ^ { - 4 }$ </td><td> $5 . 8 7 \pm 0 . 4 6 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>Euclidean</td><td> $2 . 0 3 \pm 1 . 6 8 \times 1 0 ^ { - 2 }$ </td><td> $1 . 8 7 \pm { 0 . 5 7 } \times { 1 0 ^ { - 2 } }$ </td><td> $6 . 0 0 \pm 0 . 5 7 \times 1 0 ^ { - 1 }$ </td></tr><tr><td rowspan="3">Stoch. KdV</td><td>Independent</td><td> $1 . 6 9 \pm 1 . 3 1 \times 1 0 ^ { - 3 }$ </td><td> $3 . 1 5 \pm 2 . 3 1 \times 1 0 ^ { - 3 }$ </td><td> $4 . 2 7 \pm 0 . 2 5 \times 1 0 ^ { 0 }$ </td></tr><tr><td>RBF</td><td> $1 . 1 7 \pm 0 . 7 2 \times 1 0 ^ { - 3 }$ </td><td> $1 . 6 2 \pm 0 . 9 1 \times 1 0 ^ { - 3 }$ </td><td> $4 . 0 5 \pm 0 . 1 4 \times 1 0 ^ { 0 }$ </td></tr><tr><td>Euclidean</td><td> $2 . 3 4 \pm 1 . 7 1 \times 1 0 ^ { - 3 }$ </td><td> $2 . 5 1 \pm 1 . 5 0 \times 1 0 ^ { - 3 }$ </td><td> $3 . 9 7 \pm 0 . 2 8 \times 1 0 ^ { 0 }$ </td></tr><tr><td rowspan="3">Stoch. NS</td><td>Independent</td><td> $3 . 7 2 \pm 2 . 3 1 \times 1 0 ^ { - 1 }$ </td><td> $5 . 7 6 \pm 0 . 7 6 \times 1 0 ^ { 0 }$ </td><td> $6 . 2 2 \pm 1 . 0 5 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>RBF</td><td> $3 . 3 8 \pm 1 . 1 5 \times 1 0 ^ { - 2 }$ </td><td> $2 . 5 9 \pm 0 . 7 6 \times 1 0 ^ { - 1 }$ </td><td> $3 . 2 2 \pm 0 . 6 3 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Euclidean</td><td> $1 . 8 1 \pm 0 . 6 6 \times 1 0 ^ { - }$  1</td><td> $4 . 5 9 \pm 0 . 8 1 \times 1 0 ^ { 0 }$ </td><td> $5 . 2 5 \pm 0 . 6 2 \times 1 0 ^ { - 2 }$ </td></tr></table>

## C.10 Non-Kernel Metrics

Table 3 (Section 4.1) reports sliced Wasserstein distance and marginal $W _ { 1 }$ for kFFM versus FFM on the same runs. Coupling and evaluation kernels already differ for our strongest configurations (signature coupling, RBF-MMD evaluation), and the gains persist under every non-kernel metric. Table 16 adds the autocorrelation and log-spectrum errors on the same matched runs for the sequence datasets; the pointwise diagnostics by kernel variant on the original 10-seed runs are in Appendix C.6.

Table 13: Baseline comparison for PDE datasets among models with available pointwise statistics. The full baseline roster, including $\mathrm { C F M - O T } ( L ^ { 2 } )$ , is summarized in Table 1. Following Kerrigan et al. [25], we compare kFFM against baseline models on discretization-level mean/variance MSE and log-spectrum error; the kFFM row reports the best of the kernel variants in Table 12 for each metric (an oracle over variants, used only for these diagnostics), whereas Table 1 uses the validation-selected configuration. Results are mean±std over 10 seeds. Best is green; second best is orange.
<table><tr><td>Dataset</td><td>Model</td><td>Mean</td><td>Variance</td><td>Spectrum (log)</td></tr><tr><td rowspan="4">KdV</td><td rowspan="4">DDO/NCSN DDPM GANO</td><td> $2 . 4 6 \pm 0 . 4 6 \times 1 0 ^ { - 1 }$ </td><td> $2 . 7 8 \pm 0 . 4 3 \times 1 0 ^ { - 2 }$ </td><td> $4 . 5 2 \pm 0 . 0 0 \times 1 0 ^ { 1 }$ </td></tr><tr><td> $3 . 6 7 \pm 0 . 9 1 \times 1 0 ^ { - 3 }$ </td><td> $5 . 8 1 \pm 0 . 6 9 \times 1 0 ^ { - 3 }$ </td><td> $1 . 7 1 \pm 0 . 1 3 \times 1 0 ^ { 1 }$ </td></tr><tr><td> $1 . 8 7 \pm 0 . 9 4 \times 1 0 ^ { - 1 }$ </td><td> $2 . 9 1 \pm 0 . 1 6 \times 1 0 ^ { - 2 }$ </td><td> $2 . 0 2 \pm 0 . 3 0 \times 1 0 ^ { 1 }$ </td></tr><tr><td> $5 . 2 0 \pm 3 . 3 7 \times 1 0 ^ { - 4 }$ </td><td> $1 . 3 9 \pm 0 . 4 9 \times 1 0 ^ { - 4 }$ </td><td> $1 . 7 2 \pm 0 . 0 3 \times 1 0 ^ { 1 }$ </td></tr><tr><td rowspan="5"></td><td>kFFM DDO/NCSN</td><td> $1 . 4 8 \pm 0 . 9 5 \times 1 0 ^ { - 4 }$  </td><td> $5 . 6 8 \pm 2 . 5 8 \times 1 0 ^ { - 5 }$  </td><td> $1 . 6 9 \pm 0 . 0 2 \times 1 0 ^ { 1 }$ </td></tr><tr><td>DDPM</td><td> $4 . 7 9 \pm 0 . 0 1 \times 1 0 ^ { - 1 }$ </td><td> $4 . 3 9 \pm 0 . 0 0 \times 1 0 ^ { - 2 }$ </td><td> $9 . 9 3 \pm 0 . 0 8 \times 1 0 ^ { - 1 }$ </td></tr><tr><td></td><td> $5 . 2 7 \pm 0 . 2 1 \times 1 0 ^ { - 1 }$ </td><td> $2 . 1 7 \pm 0 . 5 2 \times 1 0 ^ { - 2 }$ </td><td> $2 . 5 8 \pm 0 . 6 3 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>GANO</td><td> $6 . 1 6 \pm 4 . 2 5 \times 1 0 ^ { - 2 }$ </td><td> $2 . 7 4 \pm 1 . 5 9 \times 1 0 ^ { - 2 }$ </td><td> $2 . 2 1 \pm 0 . 1 5 \times 1 0 ^ { 0 }$ </td></tr><tr><td>FFM kFFM</td><td> $7 . 9 6 \pm 7 . 8 2 \times 1 0 ^ { - 2 }$   $2 . 0 1 \pm 2 . 0 1 \times 1 0 ^ { - 3 }$  </td><td> $3 . 1 1 \pm 0 . 8 0 \times 1 0 ^ { - 2 }$ </td><td> $7 . 9 2 \pm 0 . 8 9 \times 1 0 ^ { - 1 }$ </td></tr><tr><td rowspan="5">Stoch. KdV</td><td>DDO/NCSN</td><td> $3 . 5 9 \pm 0 . 6 0 \times 1 0 ^ { - 2 }$ </td><td> $7 . 5 1 \pm 3 . 3 3 \times 1 0 ^ { - 4 }$   $1 . 5 5 \pm 0 . 0 1 \times 1 0 ^ { - 1 }$ </td><td> $5 . 8 7 \pm 0 . 4 6 \times 1 0 ^ { - 1 }$   $1 . 8 4 \pm 0 . 0 0 \times 1 0 ^ { 1 }$ </td></tr><tr><td>DDPM</td><td> $3 . 9 0 \pm 0 . 7 8 \times 1 0 ^ { - 3 }$ </td><td> $7 . 1 9 \pm 8 . 4 8 \times 1 0 ^ { - 3 }$ </td><td> $1 . 5 7 \pm 0 . 2 2 \times 1 0 ^ { 0 }$ </td></tr><tr><td>GANO</td><td> $1 . 8 0 \pm 0 . 3 1 \times 1 0 ^ { - 3 }$ </td><td> $1 . 4 2 \pm 1 . 5 7 \times 1 0 ^ { - 2 }$ </td><td> $1 . 4 5 \pm 0 . 0 9 \times 1 0 ^ { 1 }$ </td></tr><tr><td>FFM</td><td> $1 . 6 9 \pm 1 . 3 1 \times 1 0 ^ { - 3 }$ </td><td></td><td></td></tr><tr><td>kFFM</td><td> $1 . 1 7 \pm 0 . 7 2 \times 1 0 ^ { - 3 }$  </td><td> $3 . 1 5 \pm 2 . 3 1 \times 1 0 ^ { - 3 }$ </td><td> $4 . 2 7 \pm 0 . 2 5 \times 1 0 ^ { 0 }$ </td></tr><tr><td rowspan="5">Stoch. NS</td><td>DDO/NCSN</td><td></td><td> $1 . 6 2 \pm 0 . 9 1 \times 1 0 ^ { - 3 }$  </td><td> $3 . 9 7 \pm 0 . 2 8 \times 1 0 ^ { 0 }$ </td></tr><tr><td></td><td> $3 . 4 5 \pm 0 . 0 1 \times 1 0 ^ { - 1 }$ </td><td> $7 . 5 9 \pm 0 . 0 2 \times 1 0 ^ { 0 }$ </td><td> $1 . 0 6 \pm 0 . 2 8 \times 1 0 ^ { 0 }$ </td></tr><tr><td>DDPM</td><td> $3 . 5 2 \pm 0 . 3 6 \times 1 0 ^ { - 1 }$ </td><td> $4 . 8 1 \pm 0 . 2 8 \times 1 0 ^ { 0 }$ </td><td> $8 . 6 0 \pm 2 . 3 9 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>GANO</td><td> $4 . 3 8 \pm 0 . 4 8 \times 1 0 ^ { - 1 }$ </td><td></td><td></td></tr><tr><td>FFM</td><td> $3 . 7 2 \pm 2 . 3 1 \times 1 0 ^ { - 1 }$ </td><td> $7 . 0 5 \pm 0 . 6 5 \times 1 0 ^ { 0 }$   $5 . 7 6 \pm 0 . 7 6 \times 1 0 ^ { 0 }$ </td><td> $1 . 0 3 \pm 0 . 9 3 \times 1 0 ^ { 0 }$   $6 . 2 2 \pm 1 . 0 5 \times 1 0 ^ { - 2 }$ </td></tr></table>

## C.11 Turbulent Navier–Stokes: Samples and Statistics

Table 17 gives the turbulent benchmark of Section 4.2 with explicit configurations and unscaled values, and Figure 4 shows vorticity samples (pooled vorticity PDFs and energy spectra are in Figure 1). We also re-ran the standard Navier–Stokes configurations of Section 4.1 with the same physics diagnostics (3 seeds): the flow-matching family leads (enstrophy $W _ { 1 }$ of 0.16 vs. 0.28 for DDPM; vorticity-PDF $W _ { 1 }$ of $0 . 1 4 { - } 0 . 1 5 ~ \mathrm { v s . } ~ 0 . 3 9 )$ , and kFFM matches FFM within noise on the physics metrics while retaining its distributional edge from Table 1. The kernel-cost mismatch is larger in the turbulent regime: prior-data $L ^ { 2 }$ distances more than double relative to the standard benchmark (median 91 vs. 40), pushing the RBF cost deeper into saturation, the regime where Theorem 2 is weakest; the empirical results carry the method there.

## C.12 Measuring the Kernel-Cost Mismatch

Analytically, $c _ { \mathrm { R B F } } ( f , g ) = 2 ( 1 - e ^ { - \| f - g \| ^ { 2 } / 2 \sigma ^ { 2 } } )$ is a rescaled $L ^ { 2 }$ cost up to relative error $O ( ( D / \sigma ) ^ { 2 } )$ with D the batch diameter; median normalization cancels the rescaling, so $\Delta _ { \kappa }$ is small at large bandwidth. We measured the mismatch on training-faithful batches through the rank correlation between kernel and $L ^ { 2 }$ cost matrices and the total variation (TV) distance between the induced plans. (1) Where the RBF cost is not saturated it is a monotone transform of $L ^ { 2 }$ (rank correlation ≈ 1: Gene Expression 1.000, Heston 0.98), i.e. the same geometry with compressed outliers. (2) The signature kernel genuinely re-ranks pairs (rank correlation with $L ^ { 2 }$ of only $0 . 3 3 \mathrm { - } 0 . 3 7 $ on AEMET and Gene Expression), the data class where its wins are largest and most significant; $\Delta _ { \kappa }$ is large there by design, and Theorem 2 controls the deviation from Wasserstein with the chosen cost rather than proximity to $L ^ { 2 } \mathrm { O T } .$ (3) On smooth Navier–Stokes fields the plans agree across costs (plan TV $\leq 0 . 0 2 )$ , the smal $- \Delta _ { \kappa }$ regime where the bound is tightest and the bounded cost’s stability and speed are the operative benefits.

## C.13 Synthetic Couplings: Where $L ^ { 2 }$ Pairing Fails

Figure 5 isolates the two failure modes of $L ^ { 2 }$ pairing discussed in Section 4.1 on synthetic data at the training coupling protocol (stable over 6 seeds): a conditioning failure under heavy-tailed nuisance

![](images/8c2484e809d336bbc619d71a395ed9b180aa79f7236a584ba2839fbfcb012e63.jpg)

![](images/fb4712773dee857fa4b341feeaa51702a072d0da51ac3c9d69190516971abed2.jpg)

![](images/47d9be0d7fcb9850f18edeb303ecaeb013526a9e9549d2a3c394744a62ae6beb.jpg)

![](images/6e8e2ab5f7c73abb8a0cdcd17fc8911f01486f171b53ee113bdd9c913ca58b41.jpg)

![](images/381be6e2544879463a732ddb65ee1dbd96cd8bf2cdb0522d058863010db5b99e.jpg)

![](images/8693fe68363ce1c5dd0c75b84b8378e031c916941e2fb612399c8b800801c839.jpg)

![](images/696c000b9b530cc86f92a0243fb7197ca6dda7d8b1c1e3eb970e42f889000102.jpg)

![](images/2dc0121177a2980d6351aa2f4b4336a9add43c297d9ff4d2297328575d548746.jpg)

![](images/7c966aa2e3955435dfb3400c2ed926c020fda0f832bf3b891c681f007e6e542a.jpg)

![](images/22efbb8c4467d1ae521f8f732d40b7f21653253e93cd3737ca5d66d16b72b8d7.jpg)  
(a) Samples.

![](images/6178f688ef334cc264ac40bbd0a5a54e76dd689d85f860b99926e0b636debce1.jpg)

![](images/5e6f0a4471da1cf4d8caae9f76b245ac6e1cc65b223aa57b86d6e9c04490c921.jpg)  
(b) 4x temporal super-resolution.

Figure 2: Qualitative results on AEMET; panels labeled Euclidean, RBF, and Signature are kFFM variants (kFFM-Euc, kFFM-RBF, kFFM-Sig) and Independent is FFM. Kernel OT better preserves the seasonal spread and amplitude of trajectories both at the original resolution and under temporal super-resolution.  
![](images/3c68498065a302d1fd07ba39fb947fb02b06bd3311b9e15dd7e08f76f6fa2775.jpg)  
(a) Samples.

![](images/daf800543c0241bf90dce72ff8a3d0ca4daf76d8400a3178f4f3fe7daf188257.jpg)  
(b) 2x spatial super-resolution.  
Figure 3: Qualitative results on Navier–Stokes; panels labeled Euclidean and RBF are kFFM variants (kFFM-Euc, kFFM-RBF) and Independent is FFM. Kernel OT produces smoother and more coherent flow structures than independent coupling, and this advantage persists under spatial super-resolution.

bursts, cured by any bounded cost, and a geometry failure on paths matched by volatility, cured only by the signature kernel.

## C.14 Why Straightness Arguments Do Not Directly Transfer

Finite-dimensional OT-CFM is often motivated by the Euclidean $W _ { 2 }$ picture: a better coupling, combined with linear interpolation, can lead to straighter paths and hence lower numerical integration cost. That intuition depends on geometric facts that are specific to the finite-dimensional Wasserstein setting and should not be treated as automatic in function space.

Proposition 6 (Coupling-only role of kernel OT). Fix a coupling $\pi \in \Pi ( \mu _ { 0 } , \mu _ { 1 } )$ . In kFFM, conditional on $( f _ { 0 } , f _ { 1 } ) \sim \pi$ , the path is

$$
g _ { t } ^ { f _ { 0 } , f _ { 1 } } = t f _ { 1 } + \sigma _ { t } f _ { 0 } , \qquad \sigma _ { t } = 1 - ( 1 - \sigma _ { \operatorname* { m i n } } ) t ,
$$

so that

$$
\partial _ { t } g _ { t } ^ { f _ { 0 } , f _ { 1 } } = f _ { 1 } - \left( 1 - \sigma _ { \operatorname * { m i n } } \right) f _ { 0 } , \qquad \partial _ { t } ^ { 2 } g _ { t } ^ { f _ { 0 } , f _ { 1 } } = 0 :
$$

Table 14: Higher-moment sequence-data statistics (skewness and kurtosis errors) for the models with available pointwise diagnostics, following the evaluation convention of Kerrigan et al. [25]; mean, variance, and autocorrelation errors are in Table 11, and the full baseline roster, including $\mathrm { C F M - O T } ( L ^ { 2 } )$ , in Table 1. The kFFM row uses the best kernel variant per metric. Lower is better. Best is green, second best is orange.
<table><tr><td>Dataset</td><td>Model</td><td>Skewness</td><td>Kurtosis</td></tr><tr><td rowspan="5">AEMET</td><td>DDO/NCSN</td><td> $2 . 7 3 \pm 0 . 2 4 \times 1 0 ^ { - 1 }$ </td><td> $3 . 5 7 \pm 0 . 1 8 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>DDPM</td><td> $1 . 6 1 \pm 0 . 5 0 \times 1 0 ^ { - 1 }$ </td><td> $3 . 9 5 \pm 1 . 7 9 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>GANO</td><td> $2 . 3 4 \pm 0 . 3 9 \times 1 0 ^ { - 1 }$ </td><td> $3 . 8 4 \pm 0 . 8 3 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>FFM</td><td> $5 . 5 2 \pm 4 . 0 6 \times 1 0 ^ { - 2 }$ </td><td> $3 . 9 4 \pm 4 . 1 6 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>kFFM</td><td> $8 . 1 2 \pm 5 . 3 5 \times 1 0 ^ { - 3 }$  </td><td> $4 . 2 2 \pm 2 . 6 1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td rowspan="6">Gene Expr.</td><td>DDO/NCSN</td><td> $2 . 0 2 \pm 0 . 1 3 \times 1 0 ^ { - 1 }$ </td><td> $3 . 7 5 \pm 0 . 4 4 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>DDPM</td><td> $3 . 7 2 \pm 1 . 1 7 \times 1 0 ^ { - 1 }$ </td><td> $1 . 5 7 \pm 1 . 3 1 \times 1 0 ^ { 1 }$ </td></tr><tr><td>GANO</td><td> $8 . 6 2 \pm 4 . 5 6 \times 1 0 ^ { - 1 }$ </td><td>2.88 ± 2.57 × 100</td></tr><tr><td>FFM</td><td> $1 . 7 5 \pm 0 . 3 6 \times 1 0 ^ { - 1 }$ </td><td> $1 . 6 4 \pm 0 . 5 7 \times 1 0 ^ { 0 }$ </td></tr><tr><td>kFFM</td><td> $7 . 4 3 \pm 0 . 7 6 \times 1 0 ^ { - 2 }$  </td><td> $4 . 8 7 \pm 1 . 1 7 \times { 1 0 } ^ { - 1 }$ </td></tr><tr><td>DDO/NCSN</td><td></td><td></td></tr><tr><td rowspan="5">Economy</td><td></td><td> $4 . 9 1 \pm 1 . 2 4 \times 1 0 ^ { - 1 }$ </td><td> $4 . 7 0 \pm 1 . 8 7 \times 1 0 ^ { 0 }$ </td></tr><tr><td>DDPM</td><td> $1 . 2 6 \pm 3 . 2 3 \times 1 0 ^ { 0 }$ </td><td> $9 . 0 0 \pm 3 4 . 1 0 \times 1 0 ^ { 1 }$ </td></tr><tr><td>GANO</td><td> $1 . 0 5 \pm 0 . 5 7 \times 1 0 ^ { 0 }$ </td><td> $3 . 7 8 \pm 1 . 7 1 \times 1 0 ^ { 0 }$ </td></tr><tr><td>FFM</td><td> $1 . 8 0 \pm 0 . 8 0 \times 1 0 ^ { - 1 }$ </td><td> $1 . 7 5 \pm 0 . 6 9 \times 1 0 ^ { 0 }$ </td></tr><tr><td>kFFM</td><td> $7 . 6 5 \pm 2 . 9 4 \times 1 0 ^ { - 2 }$  </td><td> $9 . 7 7 \pm 4 . 0 1 \times 1 0 ^ { - 1 }$ </td></tr><tr><td rowspan="5">Heston</td><td>DDO/NCSN</td><td></td><td> $1 . 5 8 \pm 0 . 0 0 \times 1 0 ^ { 3 }$ </td></tr><tr><td></td><td> $1 . 3 6 \pm 0 . 0 1 \times 1 0 ^ { 1 }$ </td><td></td></tr><tr><td>DDPM</td><td> $3 . 9 7 \pm 0 . 9 1 \times 1 0 ^ { 0 }$ </td><td> $9 . 5 3 \pm 2 . 0 4 \times 1 0 ^ { 2 }$ </td></tr><tr><td>GANO</td><td> $6 . 1 9 \pm 0 . 9 9 \times 1 0 _ { . } ^ { 0 }$ </td><td> $1 . 3 3 \pm 0 . 0 6 \times 1 0 ^ { 3 }$ </td></tr><tr><td>FFM kFFM</td><td> $5 . 3 0 \pm 2 . 7 9 \times 1 0 ^ { 0 }$  一  $3 . 3 1 \pm 0 . 6 1 \times 1 0 ^ { 0 }$  </td><td> $1 . 5 9 \pm 1 . 1 2 \times 1 0 ^ { 3 }$   $8 . 3 8 \pm 1 . 6 0 \times 1 0 ^ { 2 }$ </td></tr></table>

Table 15: Measured training cost of the kernel-OT coupling step across benchmarks. The $O ( b ^ { 2 } )$ Sinkhorn solve is independent of model size and resolution.
<table><tr><td>Quantity</td><td>Measurement</td></tr><tr><td>FFM</td><td>End-to-end training overhead vs. 5–7% at the largest trained resolution (1282 Navier-Stokes)</td></tr><tr><td>Peak coupling memory</td><td>80–151 MB across benchmarks and  $b ~ \in ~ \{ 6 4 , \ldots , 5 1 2 \} ; ~ 1 . 5 \mathrm { - } 9 \%$  of the FNO forward+backward peak (e.g. 116 MB vs. 7.62 GB on</td></tr><tr><td rowspan="3">Per-batch coupling time Bounded RBF vs. unbounded  $L ^ { 2 }$ </td><td>Navier-Stokes 642)</td></tr><tr><td>1–80 ms, about one FNO step 9 vs. 73 ms per batch (≈ 8×); 15–25% faster end-to-end training</td></tr><tr><td></td></tr><tr><td>Sinkhorn (b = 512) Signature-kernel memory</td><td>0.63 GB at 100 time points vs. 12.5 GB at 512; the selection rule therefore assigns  $\mathrm { R B F } { \dot { I } } L ^ { 2 }$  costs to very long sequences</td></tr><tr><td>Larger-model control</td><td>Giving FFM the +6% budget as extra epochs does not close the gap (loss curves plateau early)</td></tr></table>

the conditional path is affine in t and deterministic given the pair, and its geometry does not depend on π. Moreover, the training objective can be written as

$$
\begin{array} { r } { \mathcal { L } _ { C F M } ^ { \pi } ( \theta ) = \mathbb { E } _ { t , ( f _ { 0 } , f _ { 1 } ) \sim \pi } \left. \partial _ { t } g _ { t } ^ { f _ { 0 } , f _ { 1 } } - v _ { \theta } \left( g _ { t } ^ { f _ { 0 } , f _ { 1 } } , t \right) \right. _ { \mathcal { F } } ^ { 2 } . } \end{array}
$$

Therefore, optimizing over π changes only the distribution of endpoint pairs presented to the model;   
it does not introduce any explicit pathwise straightness, geodesic, or action-minimization term.

Proof. The path and its velocity are exactly those of Section 3.1; they coincide with the FFM path $t f + \sigma _ { t } \xi$ and conditional field $f - ( 1 - \sigma _ { \operatorname* { m i n } } ) \xi$ of Section 2.1 with the base noise $\xi$ replaced by the paired prior sample $f _ { 0 }$ . Differentiating the affine path gives the displayed derivatives. Substituting the conditional path into the conditional flow matching objective yields the displayed loss. Since π enters only through the sampling law of $( f _ { 0 } , f _ { 1 } )$ and no additional functional of the path $t \mapsto g _ { t } ^ { f _ { 0 } , f _ { 1 } }$ appears in the objective, the method does not explicitly optimize any global path-geometry quantity. □

Table 16: Non-kernel and structural metrics for kFFM-Sig vs. FFM on the sequence datasets (same prior and architecture; lower is better), computed on the matched runs of Table 2: mean±std over 10 shared seeds (Economy averages the population and GDP series within each seed). Every comparison is individually significant when paired by seed (Wilcoxon $p \leq 0 . 0 2 7$ ; kFFM better on 9–10 of the 10 seeds on both datasets); the wide autocorrelation spread on Gene Expression is across-seed variation shared by both methods, which the pairing removes. These runs are distinct from the 10-seed runs behind Table 10, so the autocorrelation values differ between the two tables; both favor kFFM.
<table><tr><td>Dataset</td><td>Metric</td><td>FFM</td><td>kFFM-Sig</td></tr><tr><td rowspan="5">Gene Expr.</td><td>Sliced-W</td><td> $0 . 1 0 6 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 0 8 7 \pm 0 . 0 0 6 }$ </td></tr><tr><td>Marginal-W1</td><td> $0 . 0 7 4 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 0 5 9 \pm 0 . 0 0 4 }$ </td></tr><tr><td>Autocorr.  $\mathrm { M S E } \left( \times 1 0 ^ { - 5 } \right)$ </td><td> $9 . 3 \pm 6 . 8$ </td><td> ${ \bf 3 . 3 \pm 3 . 6 }$ </td></tr><tr><td>log-Spectrum  $\mathrm { M S E } \left( \times 1 0 ^ { - 2 } \right)$ </td><td> $1 . 7 6 \pm 0 . 4 2$ </td><td> ${ \bf 0 . 8 6 \pm 0 . 2 2 }$ </td></tr><tr><td>Sliced-W</td><td> $0 . 0 2 9 1 \pm 0 . 0 0 1 9$ </td><td> $\mathbf { 0 . 0 2 3 3 \pm 0 . 0 0 1 2 }$ </td></tr><tr><td rowspan="4">Economy</td><td>Marginal-W1</td><td> $0 . 0 1 7 1 \pm 0 . 0 0 1 6$ </td><td> $\mathbf { 0 . 0 1 3 7 \pm 0 . 0 0 0 9 }$ </td></tr><tr><td>Autocorr. MSE  $( \times 1 0 ^ { - 4 } )$ </td><td> $1 . 5 9 \pm 0 . 3 9$ </td><td> ${ \bf 1 . 1 2 \pm 0 . 3 5 }$ </td></tr><tr><td> $\mathsf { l o g { \mathrm { - } } S p e c t r u m M S E }$ </td><td></td><td></td></tr><tr><td></td><td> $1 . 0 9 \pm 0 . 0 7$ </td><td> ${ \bf 0 . 9 7 \pm 0 . 0 4 }$ </td></tr></table>

Table 17: Full version of Table 4 with explicit configurations (cost/base measure) and unscaled values. Turbulent 2D Navier–Stokes $( \nu = 1 0 ^ { - 5 }$ , 1200 trajectories, Re ≈ 2000): distributional and physics diagnostics (lower is better), mean±std over 5 seeds, identical FNO backbone and training budget for all flow/diffusion methods. Best per column in bold (on MMD, kFFM and $\mathrm { C F M - O T } ( L ^ { 2 } )$ tie within std). $\operatorname { V o r t - P D F } W _ { 1 } \colon W _ { 1 }$ between pooled vorticity distributions; Enstrophy $W _ { 1 } \colon W _ { 1 }$ between per-snapshot enstrophy distributions; Skew/Kurt: absolute errors of pooled-vorticity skewness and excess kurtosis; log-Spec: log-spectral error.
<table><tr><td>Method</td><td>MMD</td><td>Sliced-W</td><td>Vort-PDF W1</td><td>Enstrophy W1</td><td>Skew err.</td><td>Kurt err.</td><td>log-Spec</td></tr><tr><td>kFFM (selected: Euclidean-RBF/wn) 0.00148 ± 0.0006 0.193 ± 0.01</td><td></td><td></td><td> $\mathbf { 0 . 0 3 7 5 \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 1 5 5 \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 0 0 2 7 \pm 0 . 0 0 3 }$ </td><td> $0 . 0 7 7 7 \pm 0 . 0 1$ </td><td> $\mathbf { 0 . 0 2 0 4 } \pm \mathbf { 0 . 0 0 8 }$ </td></tr><tr><td>CFM-OT(L2) (raw L2/wn)</td><td> $\mathbf { 0 . 0 0 1 4 7 \pm 0 . 0 0 1 }$ </td><td> $0 . 1 9 4 \pm 0 . 0 1$ </td><td> $0 . 0 4 0 8 \pm 0 . 0 1$ </td><td> $0 . 1 5 7 \pm 0 . 0 1$ </td><td>0.00404 ± 0.002</td><td> $0 . 0 7 5 2 \pm 0 . 0 1$ </td><td> $0 . 0 2 2 3 \pm 0 . 0 0 7$ </td></tr><tr><td>kFFM-Sob (Sobolev-RBF/gp)</td><td> $0 . 1 1 1 \pm 0 . 0 2$ </td><td> $0 . 6 1 2 \stackrel { } { \pm } 0 . 0 5$ </td><td> $0 . 1 0 4 \pm 0 . 0 3$ </td><td> $0 . 1 9 4 \pm 0 . 0 5$ </td><td> $0 . 0 4 5 6 \pm 0 . 0 2$ </td><td> $\mathbf { 0 . 0 6 7 9 \pm 0 . 0 5 }$ </td><td> $0 . 0 6 5 2 \pm 0 . 0 1$ </td></tr><tr><td>kFFM-Euc (raw  $L ^ { 2 } / \mathrm { g p } )$ </td><td> $0 . 1 0 9 \pm 0 . 0 2$ </td><td> $0 . 6 2 3 \pm 0 . 0 3$ </td><td> $0 . 1 0 4 \pm 0 . 0 4$ </td><td> $0 . 1 9 7 \pm 0 . 0 6$ </td><td> $0 . 0 4 9 5 \pm 0 . 0 3$ </td><td> $0 . 0 7 7 4 \pm 0 . 0 4$ </td><td> $0 . 0 6 7 \pm 0 . 0 1$ </td></tr><tr><td>FFM (gp, independent)</td><td> $0 . 1 0 8 \pm 0 . 0 2$ </td><td> $0 . 6 0 1 \pm 0 . 0 2$ </td><td> $0 . 1 0 7 \pm 0 . 0 4$ </td><td> $0 . 2 0 1 \pm 0 . 0 6$ </td><td> $0 . 0 4 8 7 \pm 0 . 0 3$ </td><td> $0 . 0 7 2 3 \pm 0 . 0 5$ </td><td> $0 . 0 6 4 6 \pm 0 . 0 1$ </td></tr><tr><td>DDO/NCSN</td><td> $0 . 5 6 9 \pm 0 . 0 0 9$ </td><td> $1 . 3 9 \pm 0 . 0 6$ </td><td> $1 . 2 2 \pm 0 . 0 1$ </td><td> $1 . 0 7 \pm 0 . 0 2$ </td><td> $0 . 0 0 8 3 \pm 0 . 0 1$ </td><td> $1 . 2 9 \pm 0 . 0 1$ </td><td> $0 . 9 3 7 \pm 0 . 0 2$ </td></tr><tr><td>DDPM</td><td>0.38 ± 0.02</td><td> $1 . 1 8 \pm 0 . 1$ </td><td> $0 . 8 4 8 \overset { - } { \pm } 0 . 0 3$ </td><td> $0 . 8 8 9 \pm 0 . 0 2$ </td><td> $0 . 0 7 8 9 \pm 0 . 0 5$ </td><td> $3 . 3 7 \pm 2$ </td><td> $0 . 1 1 7 \pm 0 . 0 3$ </td></tr></table>

Corollary 2 (Coupling-invariance of ambient path straightness). For a twice differentiable path $m : { [ 0 , 1 ] } \stackrel { \cdot } {  } \mathcal { F }$ , define the ambient acceleration and speed-variationfunctionals

$$
\mathsf { A } ( m ) : = \int _ { 0 } ^ { 1 } \| \partial _ { t } ^ { 2 } m _ { t } \| _ { \mathcal { F } } ^ { 2 } d t , \qquad \mathsf { V } ( m ) : = \int _ { 0 } ^ { 1 } \bigg ( \| \partial _ { t } m _ { t } \| _ { \mathcal { F } } - \int _ { 0 } ^ { 1 } \| \partial _ { s } m _ { s } \| _ { \mathcal { F } } d s \bigg ) ^ { 2 } d t .
$$

Then for every coupling $\pi \in \Pi ( \mu _ { 0 } , \mu _ { 1 } )$ and every endpoint pair $( f _ { 0 } , f _ { 1 } ) \sim \pi ,$ the kFFM conditional path $g ^ { f _ { 0 } , f _ { 1 } }$ satisfies

$$
\mathsf { A } ( g ^ { f _ { 0 } , f _ { 1 } } ) = 0 , \qquad \mathsf { V } ( g ^ { f _ { 0 } , f _ { 1 } } ) = 0 .
$$

Hence these natural ambient-space straightness measures of the prescribed conditional path are identicalfor all couplings.

Proof. By Proposition $6 , \partial _ { t } g _ { t } ^ { f _ { 0 } , f _ { 1 } } = f _ { 1 } - ( 1 - \sigma _ { \operatorname* { m i n } } ) f _ { 0 }$ is constant in t and $\partial _ { t } ^ { 2 } g _ { t } ^ { f _ { 0 } , f _ { 1 } } = 0$ . Therefore $\mathsf { A } ( g ^ { f _ { 0 } , f _ { 1 } } ) = 0$ . The speed $\| \partial _ { t } g _ { t } ^ { f _ { 0 } , f _ { 1 } } \| _ { \mathcal { F } } = \| f _ { 1 } - ( 1 - \sigma _ { \operatorname* { m i n } } ) f _ { 0 } \| _ { \mathcal { F } }$ is also constant in t, so it coincides with its own time average and $\mathsf { V } ( g ^ { f _ { 0 } , f _ { 1 } } ) = 0$ □

Proposition 6 and Corollary 2 formalize the internal reason that the usual OT-CFM straightness argument does not transfer directly to kFFM: kernel OT acts only at the level of endpoint coupling, while the interpolation template remains the same affine FFM path. Any improvement in convergence or sampling cost could therefore only arise indirectly through better endpoint pairings and the learned velocity field, not from an explicit path-straightness objective, and we do not claim one.

Our setting is different in a second, geometric way. FFM is formulated on Gaussian measures over separable Hilbert spaces, where absolute continuity and transport already depend on Cameron–Martin and Feldman–Hajek type conditions rather than Lebesgue-density arguments. Even when one restricts attention to Gaussian optimal transport, the appropriate geometry is the Bures–Wasserstein geometry rather than the flat Euclidean geometry used by standard OT-CFM heuristics.

![](images/4f503c907e9449b1bc0d4850441e4760d9b1ac4a595bf6552903533734a90742.jpg)  
Figure 4: Turbulent 2D Navier–Stokes $( \nu = 1 0 ^ { - 5 } ) :$ vorticity samples on a shared color scale (for each method, the samples whose enstrophy is closest to the data median). kFFM (selected) and FFM reproduce the data’s large-scale vortical organization, DDO/NCSN produces near-noise fields, and DDPM coarse blobs. The corresponding pooled vorticity PDFs and energy spectra are in Figure 1.

Recent work by Yun and Zemel [52] makes this distinction explicit. For Gaussian measures on separable Hilbert spaces, they show that the optimal transport map can fail to admit the usual Brenier variational interpretation: formally, it is the subgradient of a convex function that is infinite almost everywhere, and the geodesic structure between degenerate Gaussian measures is richer than the classical finite-dimensional McCann interpolation picture. This does not mean that efficient flows are impossible in function space; rather, it means that the usual “OT implies straighter paths” argument is not a theorem in the present setting.

For this reason, our paper does not claim that kFFM is geodesic or pathwise optimal in a Bures– Wasserstein sense. The theoretical guarantees in this paper concern the well-posedness, error decomposition, and discretization invariance of the kernel OT surrogate, not global straightness of the learned ODE.

This is also why Section 4 makes no efficiency claim: the main supported claim of the paper is better distributional matching from geometry-aware coupling, and Appendix C.15 reports only the training-time convergence of the target metric.

## C.15 Target-Metric Convergence During Training

In finite-dimensional OT-CFM, optimal transport is often motivated by improved convergence and straighter flows [48]. As discussed in Appendix C.14, we do not treat such properties as theoretical consequences in function space and make no efficiency claim. We report only the convergence of the target metric during training: Table 18 and Figure 6 show the MMD-RBF of FFM and kFFM over the final phase of training, with both models sharing the GP prior and FNO backbone so that the only change is the coupling. The curves are the best-so-far value (cumulative minimum) of the across-seed mean; on KdV, where the unbiased estimate is noisy at the level of single checkpoints, the curves are instead the running mean over the final phase. kFFM ends training at a better MMD-RBF on every dataset, with the largest gains on Stoch. KdV, Gene Expr., AEMET, Economy, and Heston.

![](images/0802ac21a6ed4a3305b2c970c648dfa465c1fad303a2bb9a64d561dfa161db3e.jpg)  
Figure 5: Entropic minibatch couplings $( b = 6 4 , \epsilon = 0 . 0 1$ , median-normalized costs as in training) for three synthetic scenarios (rows) and four costs (columns). Sources and targets are sorted so that the intended matching is the diagonal, and each panel reports the plan mass on intended matches (a uniform plan gives ≈ 0.14–0.19). Top: shifted bumps matched by location; every cost recovers the matching. Middle: bumps where 40% of samples carry large, independent nuisance bursts. Burst pairs dominate the median normalization of the unbounded $\breve { L } ^ { 2 }$ cost, collapsing clean-pair contrasts below $\epsilon ,$ so its plan degenerates to near uniform (0.16); the bounded RBF cost saturates the bursts and recovers the matching (0.58), invariant to burst amplitude (the conditioning effect of Theorem 1). Bottom: Brownian paths $f = v W ( t ) , v \sim U [ 0 . 5 , 2 ]$ , matched by volatility. The expected $L ^ { 2 }$ cost is separable, E $| f - { \bar { g } } | | ^ { 2 } = ( v ^ { 2 } + { \dot { w } } ^ { 2 } ) \int t d t .$ , so it carries no assortative signal in expectation and its plan stays near uniform (0.25); the signature kernel (lead-lag) reads volatility through quadratic variation and concentrates on the intended matching (0.43). No single cost wins every scenario, which motivates the per-data-class selection rule of Appendix C.5.

Table 18: Relative MMD-RBF improvement of kFFM over FFM at the end of the convergence curves of Figure 6 (higher is better; values above 100% occur where the unbiased estimate for kFFM is negative). Both methods share the GP prior and FNO backbone, isolating the coupling as the only changed component.
<table><tr><td>Dataset</td><td>Heston</td><td>AEMET</td><td>KdV</td><td>Economy</td><td>Gene Expr.</td><td>Stoch. KdV</td><td>Stoch. NS</td><td>Navier-Stokes</td></tr><tr><td>∆MMD-RBF (%)</td><td>+32%</td><td>+53%</td><td>+8%</td><td>+52%</td><td>+64%</td><td>+225%</td><td>+7%</td><td>+9%</td></tr></table>

## C.16 Hyperparameter Sensitivity

Unless otherwise noted, we keep the training setup, neural architectures, Gaussian prior, and remaining OT settings fixed at the default values from the main experiments, and vary only one hyperparameter at a time. We use the $\mathtt { P 0 T } ^ { 6 }$ package [17] for optimal transport. The figures below are direct one-at-a-time sweeps: they vary σ or ϵ while holding the other settings fixed. We additionally report the sensitivity to the Gaussian-prior length scale on the 2D PDE datasets, where it has the clearest effect.

![](images/9336b3dab1d6540fa4e85bbda718f83ccdcaf326dd3c5774d0f2e5ae9fa9612f.jpg)  
Figure 6: Per-dataset target-metric convergence using MMD-RBF (lower is better) corresponding to the headline summary in Table 18. The plot shows the best-so-far across-seed mean MMD-RBF over the final phase of training (on KdV, the running mean over the final phase, from epoch 125) for kFFM versus the original FFM under matched GP prior and FNO backbone, so the only change is the coupling. The kFFM curve uses the signature kernel on Stoch. KdV, Gene Expr., AEMET, Economy, and Heston, the Sobolev-RBF cost on KdV, the Euclidean–RBF cost on Stoch. NS, and the raw L<sup>2</sup> cost on Navier–Stokes and Heston-Long. kFFM ends training at a better MMD-RBF on every dataset, with especially large gains on Stoch. KdV, Gene Expr., AEMET, Economy, and Heston. The additional Heston-Long panel is a long-horizon variant of Heston (1000 time steps) used only in this diagnostic and in the sensitivity study; it is not one of the eight benchmark datasets.

The kernel-specific hyperparameters are:

1. RBF kernel: the bandwidth σ ∈ {0.1, 0.2, 0.5, 1, 2, 5, 10}.

2. Signature kernel:

(a) Boolean switches for the lead-lag and time augmentations (lead\_lag, time\_aug).

(b) Dyadic order in {0, 1, 2, 3}.

(c) Static-kernel bandwidth between 0.1 and 5.

(d) Maximum sequence length (subsampling) in {32, 50, 64, 128}.

The hyperparameters for the Sinkhorn algorithm are:

1. ot\_reg, the regularization parameter ϵ; the sweeps of Figure 8 use {0.01, 0.05, 0.1, 0.5, 1.0}, and the full grids extend this range.

2. ot\_method: Sinkhorn or exact OT; Sinkhorn generally performs better.

![](images/dde0a94684da177459db52877b7315ed871e5140421fb171426525d4a74c6c89.jpg)  
Figure 7: Aggregate MMD-RBF sensitivity of kFFM hyperparameters across datasets; the spread is $\mathrm { ( m a x - m i n ) } / | \mathrm { m e a n } |$ over the tested settings of the hyperparameter in parentheses, and lower is less sensitive. Signature-kernel settings are nearly invariant on the tested sequence datasets, the Sinkhorn regularization is typically less influential than the RBF bandwidth, and the results depend more on ϵ under the raw $L ^ { 2 }$ (Euclidean) cost than under the bounded RBF cost (median spread of about 8% vs. near zero).

3. ot\_coupling: sampling from the plan (default) or barycentric projection; sampling usually performs better.

Figure 7 summarizes the aggregate MMD-RBF spread across datasets for the main tunable kernel families, where the spread for a given dataset is computed as (max − min)/|mean| over the tested settings. The dominant pattern is that the Sinkhorn regularization is typically quite stable, while the RBF bandwidth can matter more on selected datasets. Figure 8 then shows the direct sweeps on representative datasets, run at the white-noise base measure: the top panel varies the RBF bandwidth, and the bottom panel varies the Sinkhorn regularization. We focus these direct sweeps on the axes that showed the most variation; the signature-kernel settings were substantially flatter on the tested sequence datasets and are summarized in Figure 7. Re-aggregating the full grids: results are indistinguishable within seed-to-seed variation for RBF bandwidths $\sigma \in [ 0 . 1 , 2 ]$ on every dataset (the wider sweeps in Figure 8 show where larger bandwidths start to matter on selected datasets); the Sinkhorn regularization is flat across three orders of magnitude $( \epsilon \in [ 0 . 0 0 1 , 1 ] )$ for bounded costs, whereas with the raw $L ^ { 2 } \cos \epsilon \geq 0 . 5$ degrades long volatility paths by ≈ 10×; and the signature hyperparameters (dyadic order $0 { - } 3 .$ static bandwidth 0.1–5) leave the results unchanged. The kernel family and the base measure are the choices that matter: Euclidean costs, raw or bounded, still lose to the signature kernel by 2.6–2.9× on Gene Expression and Economy (Section 4.1).

The choice of Gaussian prior matters more on the 2D PDE datasets (Navier–Stokes and Stoch. NS), independently of the kernel. Figure 9 shows the sensitivity of these two datasets to the length scale of the Gaussian prior, which encodes a prior belief about the smoothness of the data.

## D Extended Related Work

This section expands the discussion of Section 5. The study of using optimal transport [49] to improve flow matching was first proposed by Pooladian et al. [39], Tong et al. [48], which add an OT-induced coupling of each minibatch before the interpolation and training step. Several follow-up works [28, 51] have aimed to further expand the framework. All have assumed that the sample space is $\mathbb { R } ^ { d }$ In order to better capture the resolution-invariance of grid data and continuity of path data, Kerrigan et al. [25] first proposed Functional Flow Matching, a framework to perform flow matching in Hilbert space, relying on Gaussian measure theory. However, OT-based endpoint coupling had not been brought to this space.

The line of research that embeds probability measures in RKHS has been proposed by the pioneering work of Gretton et al. [20], Sriperumbudur et al. [46], where kernel-induced metrics, such as Integral Probability Metrics (IPM), have been studied extensively [10, 46] and used for two-sample hypothesis testing [20], most recently in path space [6] with the signature kernel. The IPM in the form of the Maximum Mean Discrepancy (MMD) has been linked with both probability divergences and

![](images/db84e636fc2a0638bf01e906acb8260e93ac703c766993efb8d76342086d75c8.jpg)  
Figure 8: Direct one-at-a-time MMD-RBF sweeps on representative datasets (mean and one standard deviation across seeds). Top: varying the RBF bandwidth σ at fixed Sinkhorn regularization $( \epsilon = 0 . 1 )$ Bottom: varying ϵ at fixed bandwidth. All arms in this figure use the white-noise base measure: the solid curve (legend: CFM-OT(RBF)) is the Euclidean–RBF kernel coupling, the dashed line is $\mathrm { C F M - O T } ( L ^ { 2 } )$ , and the dotted line (legend: No coupling) is independent coupling at the same base. Across the tested values the swept arm moves within its seed-to-seed variation.

![](images/4e89ec224085f6d91e3eed72c354a859ce3caa990edee760872e9ec26d237b90.jpg)  
Figure 9: Distribution of the pointwise metrics (mean MSE, variance MSE, log-spectrum error) on the two 2D PDE datasets, Navier–Stokes and Stoch. NS, as a function of the length scale of the Gaussian prior.

Wasserstein distances [16], and it has an entropic OT formulation in the Hilbert Sinkhorn Divergence of Li et al. [31]. It has also been used as the objective of gradient flows [1, 18], which have been adopted in deep generative models. The extension of these frameworks to function space is an active area of research, and to our knowledge our work is the first to explore it in the context of flow matching.

Concurrent function-space flow matching. Three concurrent works are closest to ours; none studies the endpoint coupling. Kollovieh et al. [27] use flow matching with GP priors for probabilistic time-series forecasting, a conditional task with an independent prior-data coupling; this is closest in spirit to our GP-prior setup and orthogonal on the axis we study, and our results suggest that framework could itself benefit from kernel-OT coupling. Zhang and Scott [53] straighten flows in Hilbert space via rectification, iteratively re-training on model-generated endpoint pairs; rectification changes pairs across training rounds using the model, whereas kernel OT changes pairs within each batch using data geometry in a single run and extends beyond Hilbert spaces (signature kernel on path space), so the two can in principle be composed. Lee and Flouris [30] propose operator flow matching for time-series forecasting, again conditional prediction with independent coupling. On the finite-dimensional side, Fatras et al. [13, 14] analyze minibatch OT plans and the expected-minibatch coupling that Section 3.1 relies on.