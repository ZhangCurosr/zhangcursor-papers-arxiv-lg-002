# WHEN RIEMANN FLOWS WITH WASSERSTEIN: GENERATIVE MODELING OF PROBABILITY DISTRIBUTIONS ON MANIFOLDS

DORON HAVIV<sup>∗,1</sup> , EDWARD DE BROUWER<sup>∗,1</sup> , RISHABH ANAND<sup>2</sup> , REX YING<sup>2</sup> AÏCHA BENTAIEB<sup>1</sup> , GABRIELE SCALIA<sup>1</sup> , AND HECTOR CORRADA BRAVO<sup>1</sup>

Abstract. Many scientific datasets, such as molecular conformational ensembles or single-cell tissue measurements, are naturally modeled as meta-distributions: distributions over probability measures on non-Euclidean domains. Existing generative methods largely assume Euclidean geometry and fail to capture this structure. We introduce Riemannian Wasserstein Entropic Flow Matching (RWEFM), a generative framework on the Wasserstein space $\mathcal { P } _ { 2 } ( \mathcal { M } )$ of a Riemannian manifold $( \mathcal { M } , g )$ RWEFM is trained by regressing a neural vector field onto Riemannian optimal transport velocities, using McCann displacement interpolations as conditional paths. We confirm theoretically that this construction leads to a valid flow matching approach on $\mathcal { P } _ { 2 } ( \mathcal { M } )$ and introduce the Riemannian Entropic Map, a GPU-eficient approximation of the optimal transport map on manifolds. Our experiments show that by respecting the intrinsic geometry of the data, RWEFM can generate whole single-cell samples in hyperspherical latent spaces and protein conformational ensembles on the torus. As RWEFM requires only a geodesic distance and a projection operator, it is not restricted to manifolds with closed-form geometry, which we demonstrate by generating distributions on a general triangulated mesh. Code and tutorials are available at RWEFM.

1. Introduction. Many modern scientific datasets are most naturally represented as collections of distributions of data points living on structured non-Euclidean spaces. Molecular conformational ensembles [Axelrod and Gomez-Bombarelli, 2022] describe each molecule as a distribution over rotations and translations in 3D space; climate records [Abatzoglou et al., 2018] can be viewed as distributions of weather variables over the sphere; and tissue-level single-cell $\mathrm { R N A - s e q }$ [CZI Cell Science Program et al., 2025] represents each sample as an empirical distribution of cell states living on a biologically meaningful manifold (e.g. spherical or hyperbolic). In these settings, the object of interest is a distribution over probability measures on a manifold.

Generative modeling provides a principled way to summarize such data and to enable scientific tasks, including in-silico design [Gruver et al., 2023], hypothesis testing [Candes et al., 2018], and causal discovery in the underlying physical process [Zhu et al., 2023]. While recent generative models have achieved impressive results in Euclidean data (images [Lin et al., 2024], videos [Jin et al., 2025], single-cell [Klein et al., 2025]), and have increasingly incorporated non-Euclidean structures (e.g. molecules [Schneuing et al., 2024]), most of this progress concerns generating individual data points defined as finite-dimensional vectors. In contrast, generative modeling of distributions requires operating in an infinite-dimensional space: the Wasserstein space of probability measures $\mathcal { P } _ { 2 } ( \mathcal { M } )$ . Recent work has explored generative models directly on Wasserstein space [Haviv et al., 2024, Atanackovic et al., 2024, Piening et al., 2026] but these methods assume the underlying domain to be Euclidean $( i . e . \ M \equiv \mathbb { R } ^ { d } )$ thereby failing to faithfully model datasets whose support lies on curved geometries.

In this work, we introduce Riemannian Wasserstein Entropic Flow Matching (RWEFM), a flow-matching (FM) framework [Lipman et al., 2022] for generative modeling on $\mathcal { P } _ { 2 } ( \mathcal { M } )$ . RWEFM lifts conditional FM to the Wasserstein space using McCann displacement interpolations as conditional paths between paired measures $( \mu , \nu )$ . We establish that this yields a valid FM objective on $\mathcal { P } _ { 2 } ( \mathcal { M } )$ and that the induced marginal dynamics satisfy a weak continuity equation and admit a path-space representation via the superposition principle [Pinzi and Savaré, 2025].

A key practical bottleneck is the computation of optimal transport maps between empirical distributions on Riemannian manifolds. To this end, we introduce the Riemannian Entropic Map, a GPU-eficient approximation of the Monge map that extends the entropic map [Pooladian and Niles-Weed, 2021] to general geometries. In essence, our estimator is obtained as the exponential map of the barycentric projection of tangent vectors. We further provide an error bound that quantifies the approximation induced by regularization and finite sampling.

We evaluate RWEFM across a range of geometries and scientific applications. This includes generation of digits, letters and Kanji characters on the sphere $\mathbb { S } ^ { 2 }$ , hyperbolic disk $\mathbb { H } ^ { 2 }$ and torus $\mathbb { T } ^ { 2 } .$ , as well as whole single-cell sample generation in hyperspheri-

![](images/dcb296c5e8e8b0bda167facbdc6c711d1fadc09366640da1900cf080607072ed.jpg)  
Fig. 1: Riemannian Wasserstein Entropic Flow Matching. Top: Base Riemannian manifold $( \mathcal { M } , d _ { g } )$ Each blob is a probability distribution $\mu , \nu \in$ $\mathcal { P } _ { 2 } ( \mathcal { M } )$ and a single training example is one blob, not one point. The dashed line is the Riemannian OT path between $\mu$ and ν which is estimated via pointcloud samples from each blob using the entropic map. Bottom: Wasserstein manifold $( \mathcal { P } _ { 2 } ( \mathcal { M } ) , \mathcal { W } _ { 2 } )$ , where a blob is a distribution-over-distributions and each point is a distribution in the top figure. The same dashed path lifts here as the McCann interpolant $\mu _ { t } ,$ , the geodesic along which RWEFM learns its flow.

cal latent spaces $\bar { \mathbb { S } } ^ { 1 2 8 - 1 }$ and protein conformational ensembles on $\mathbb { T } ^ { 2 }$ . To emphasize that our framework is not limited to closed-form geometries, we further generate distributions on a general triangulated mesh. Our experiments show that respecting the intrinsic geometry of the data and using the Riemannian Entropic Map improves generation quality.

Contributions. (i) We propose RWEFM, a framework for generative modeling on the space of probability distributions on Riemannian manifolds—including general geometries without closed-form geodesics, such as triangulated meshes—and show that the resulting marginal dynamics are theoretically well-posed. (ii) We introduce the Riemannian Entropic Map, a fast and accurate approximation of Riemannian optimal transport maps suitable for large-scale GPU training, together with theoretical guarantees. (iii) We benchmark RWEFM against FM baselines on real-world datasets such as generation of single-cell samples and protein conformational ensembles.

## 2. Background and Related Work.

2.1. Optimal Transport on Riemannian Manifolds. Optimal Transport (OT) [Villani, 2008] provides a geometric framework for comparing probability distributions on Riemannian manifolds. Let $( \mathcal { M } , g )$ be a complete Riemannian manifold equipped with a metric $g$ and its corresponding geodesic distance $d _ { g } ( \cdot , \cdot )$ , and let $\mu , \nu \in \mathscr { P } _ { 2 } ( \mathcal { M } )$ be two probability measures with finite second moments. The Monge formulation seeks a deterministic map $T : { \mathcal { M } } \to { \mathcal { M } }$ pushing $\mu$ onto $\nu$ (denoted $T _ { \# } \mu = \nu )$ that minimizes the total transportation cost:

$$
\operatorname* { i n f } _ { T } \left\{ \int _ { \mathcal { M } } c ( x , T ( x ) ) d \mu ( x ) : T _ { \# } \mu = \nu \right\} ,\tag{2.1}
$$

where $\begin{array} { r } { c ( x , y ) = \frac { 1 } { 2 } d _ { g } ^ { 2 } ( x , y ) } \end{array}$ is the cost of moving a unit of mass from $x$ to $y .$ Given mild assumptions, such as $\mu$ being absolutely continuous with respect to the volume measure, the Brenier–McCann theorem [Ambrosio et al., 2021] states that there exists a unique optimal transport map $T _ { 0 }$ , known as the Monge map, of the form $T _ { 0 } ( x ) = \exp _ { x } ( - \nabla \varphi _ { 0 } ( x ) )$ ) where $\varphi _ { 0 } : { \mathcal { M } } \to$ R is a c-concave function.

The Kantorovich formulation relaxes the Monge problem by allowing for probabilistic couplings:

$$
W _ { 2 } ^ { 2 } ( \mu , \nu ) = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int _ { \mathcal { M } \times \mathcal { M } } d _ { g } ^ { 2 } ( x , y ) d \pi ( x , y ) .
$$

where $\Pi ( \mu , \nu )$ is the set of joint distributions on ${ \mathcal { M } } \times { \mathcal { M } }$ with marginals $\mu$ and $\nu .$ The optimal value $W _ { 2 } ( \mu , \nu )$ defines the 2-Wasserstein distance between the measures. Unlike the Monge formulation, the Kantorovich problem does not require continuity of measures to admit a solution. When the Monge map $T _ { 0 }$ exists, it can be recovered from the Kantorovich problem via the dual formulation: $\begin{array} { r } { \frac { 1 } { 2 } W _ { 2 } ^ { 2 } ( \mu , \nu ) = \operatorname* { s u p } _ { \varphi } \int _ { \mathcal { M } } \varphi d \mu + } \end{array}$ $\begin{array} { r } { \int _ { \mathcal { M } } \varphi ^ { c } d \nu , } \end{array}$ with $\begin{array} { r } { \varphi ^ { c } ( y ) = \operatorname* { i n f } _ { x \in \mathcal { M } } \{ \frac { 1 } { 2 } d ( x , y ) ^ { 2 } - \varphi ( x ) \} } \end{array}$ , and the Monge map is then $T _ { 0 } ( x ) =$ $\mathrm { e x p } _ { x } ( - \nabla \varphi _ { 0 } ( x ) )$ .

2.1.1. Statistical Estimation of Optimal Transport. Closed-form solutions for optimal transport maps are typically limited to very simple distributions so we generally rely on statistical estimation from finite samples $\{ x _ { i } \} _ { i = 1 } ^ { m } \sim \mu$ and $\{ y _ { j } \} _ { j = 1 } ^ { n } \sim \nu .$ However, the Kantorovich problem on discrete samples is a linear program with cubic complexity in the number of samples, hindering its application to large datasets. Instead, practical approaches rely on entropic OT, which regularizes the (discrete) objective with an entropy term $H ( \pi )$ . The problem is strictly convex and can be eficiently solved using the Sinkhorn algorithm [Cuturi, 2013].

$$
\operatorname* { m i n } _ { \pi \in \Pi ( \hat { \mu } , \hat { \nu } ) } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } c ( x _ { i } , y _ { j } ) \pi _ { i j } + \varepsilon \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } \pi _ { i j } \log \pi _ { i j } .
$$

This problem admits the dual formulation:

$$
\operatorname* { s u p } _ { f \in L ^ { 1 } ( \mu ) } \sum _ { i } f ( x _ { i } ) + \sum _ { j } g ( y _ { j } ) - \varepsilon \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } e ^ { \left( f ( x _ { i } ) + g ( y _ { j } ) - { \frac { 1 } { 2 } } d _ { g } ( x _ { i } , y _ { j } ) ^ { 2 } \right) / \varepsilon } + \varepsilon\tag{2.2}
$$

While entropic optimal transport is widely used for computing the optimal transport distances, Pooladian and Niles-Weed [2021] showed that it can be used to compute a tractable estimator of the Monge map $T _ { 0 }$ . Given the optimal entropic coupling $\pi _ { \varepsilon } .$ the entropic map is the barycentric projection $T _ { \varepsilon } ( x _ { i } ) = \mathbb { E } _ { \pi _ { \varepsilon } } [ Y | X = x _ { i } ]$ which, under suitable regularity assumptions and an appropriate joint choice of regularization and sample size, consistently estimates the Monge map $T _ { 0 }$ . The optimal entropic potentials $( f _ { \varepsilon } , g _ { \varepsilon } )$ are the solutions of Eq (2.2).

2.1.2. Wasserstein Geometry. The Wasserstein space over a Riemannian manifold $( \mathcal { M } , g )$ , denoted $\mathcal { P } _ { 2 } ( \mathcal { M } )$ , is the space of probability measures on $\mathcal { M }$ with finite second moments, equipped with the 2-Wasserstein distance $W _ { 2 }$ . While not rigorously a Riemannian manifold due to infinite dimensionality, it can be endowed with a Riemannian-like structure. There, the tangent space at a measure $\mu \in \mathcal P _ { 2 } ( \mathcal M )$ is identified with the closure of the set of gradients of smooth functions in $L ^ { 2 } ( \mu , T \mathcal { M } )$

$$
T _ { \mu } \mathcal { P } _ { 2 } ( \mathcal { M } ) = \overline { { \left\{ v = \nabla \phi : \phi \in C _ { c } ^ { \infty } ( \mathcal { M } ) \right\} } } ^ { L ^ { 2 } ( \mu ) } ,
$$

and endowed with the norm $\begin{array} { r } { \| v \| _ { L ^ { 2 } ( \mu ) } ^ { 2 } = \int _ { \mathcal { M } } \| v \| _ { g } ^ { 2 } d \mu ( x ) } \end{array}$ . The exponential and logarithm maps in this space are: $\begin{array} { r } { \exp _ { \mu } ( v ) : = \exp _ { x } ( v ( x ) ) _ { \# } \mu , \log _ { \mu } ( \nu ) : = - \nabla \varphi ( x ) } \end{array}$ , where $T ^ { \mu  \nu } =$ $\exp _ { x } ( - \nabla \varphi ( x ) )$ is the optimal transport map from $\mu$ to ν. Geodesics in $\mathcal { P } _ { 2 } ( \mathcal { M } )$ between measures $\mu$ and $\nu$ are probability paths $( \mu _ { t } ) _ { t \in [ 0 , 1 ] }$ where:

$$
\mu _ { t } = \exp _ { \mu } ( t \log _ { \mu } ( \nu ) ) = \exp _ { x } ( - t \nabla \varphi ( x ) ) _ { \sharp } \mu
$$

2.2. Flow Matching on Riemannian Manifolds and the Wasserstein Space. Flow matching generates samples from a target distribution $\nu$ from samples of a source distribution $\mu$ by learning a time-dependent vector field that generates a probability path $\mu _ { t }$ such that $\mu _ { 0 } = \mu$ and $\mu _ { 1 } = \nu$ . Directly learning such a vector field is generally impossible. Instead, flow matching defines conditional probability paths $\mu _ { t } ( \cdot | z )$ for which computing a generating vector field is tractable.

Ingredients of flow matching. This construction relies on three key components: (C1) a condition for when a vector field generates a probability path, (C2) the marginalization of the conditional vector field generates the marginal probability path, and (C3) the losses from regressing the learnable vector field to the conditional or the marginal vector field are equivalent. The continuity equation on $( \mathcal { M } , g )$ $\partial _ { t } \mu _ { t } ( x ) + \nabla _ { g } \cdot ( \mu _ { t } ( x ) v _ { t } ( x ) ) = 0$ , plays the role of the first component. The second is satisfied with the marginal vector field defined as:

$$
v _ { t } ( x ) = \int v _ { t } ( x | z ) \frac { \mu _ { t } ( x | z ) } { \mu _ { t } ( x ) } d \pi ( z ) ,
$$

Indeed, if $( \mu _ { t } ( \cdot | z ) , v _ { t } ( \cdot | z ) )$ solves the continuity equation, $\left( \mu _ { t } , v _ { t } \right)$ is also a solution. We use $z = ( x _ { 0 } , x _ { 1 } ) \sim \pi$ and π is a coupling from $\mu$ and $\nu ,$ for which multiple options have been identified in the literature [Pooladian et al., 2023, Lipman et al., 2022, Tong et al., 2023]. A typical conditional path is then a dirac centered on a curve $x _ { t } ( x _ { 0 } , x _ { 1 } )$ that interpolates between $x _ { 0 }$ and $x _ { 1 }$ . In this case, the generating vector field is given by $\begin{array} { r } { v _ { t } ( x | z ) = \dot { x } _ { t } = \frac { d } { d t } x _ { t } ( x _ { 0 } , x _ { 1 } ) } \end{array}$ . Finally, the equivalence of losses (C3) follows from the second point, as shown in [Lipman et al., 2022] (see Lipman et al. [2024] for a comprehensive tutorial). This leads to the celebrated flow matching objective:

$$
\mathbb { E } _ { t \sim \mathcal { U } [ 0 , 1 ] , ( x _ { 0 } , x _ { 1 } ) \sim \pi , x _ { t } \sim \mu _ { t } ( \cdot | z ) } \left[ \| v _ { \theta } ( x _ { t } , t ) - \dot { x } _ { t } \| _ { g ( x _ { t } ) } ^ { 2 } \right]
$$

Recently Haviv et al. [2024] extended this framework to the Wasserstein space over Euclidean domains, termed Wasserstein Flow Matching (WFM). They consider distributions on the Wasserstein space (i.e. probability distribution on the space of probability distributions), $\mathbb { P } _ { 0 } , \mathbb { P } _ { 1 } \in \mathcal { P } _ { 2 } ( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) )$ . In practice, samples from $\mathbb { P } _ { 0 }$ and $\mathbb { P } _ { 1 }$ are represented as empirical distributions or point clouds $( X _ { 0 } , X _ { 1 } )$ and $X _ { t }$ is defined as the McCann interpolant: $X _ { t } = ( 1 - t ) X _ { 0 } + t \hat { T } ^ { X _ { 0 }  X _ { 1 } } ( X _ { 0 } )$ , with ${ \hat { T } } ^ { X _ { 0 }  X _ { 1 } }$ the optimal transport map between X and $X _ { 1 }$ . One then learns the target vector field $u _ { \theta } ( X _ { t } , t ) \in T \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ by regressing it against $\dot { X } _ { t } = \hat { T } ^ { X _ { 0 }  X _ { 1 } } ( X _ { 0 } ) - X _ { 0 }$

![](images/078f97640b7827cae1706ea26422fb14cd6f535764c0e6835a04894993f5cc35.jpg)  
Fig. 2: Generation of distributions on non-Euclidean spaces Examples of empirical distributions generated by the proposed Riemannian Wasserstein Entropic Flow Matching model on non-Euclidean geometries. The top row displays MNIST digits (3, 4, and 8) generated on the sphere $\mathbb { S } ^ { 2 }$ shown in 2D via Mollweide projections. The bottom row shows EMNIST characters (H, W, and Y) generated on the hyperbolic plane $\mathbb { H } ^ { 2 }$ , visualized using the Poincaré disk model.

## 2.3. Related Work.

Optimal Transport on Riemannian Manifolds.. Optimal Transport on Riemannian manifolds has been extensively studied in the mathematical literature, with founda tional results on the existence and uniqueness of Monge maps, duality theory, and regularity properties [McCann, 2001, Villani, 2008]. These theoretical insights have paved the way for many applications, particularly in generative modeling for non-Euclidean data [Bose et al., 2023, De Bortoli et al., 2022, Huguet et al., 2023]. Relatedly, You [2026] study intrinsic and tangential barycentric projections of transport plans on Riemannian manifolds. Recent works have also explored Optimal Transport on Wasserstein spaces themselves, proving existence of continuity equations and geodesics despite the infinite dimensionality, enriching the theoretical framework and expanding its applicability [Bonet et al., 2025, Emami and Pass, 2025, Pinzi and Savaré, 2025].

Generative Modeling of Distributions. Our work relates to recent efforts in defining generative models over spaces of probability measures. Fisher FM [Davis et al., 2024] and Categorical FM [Cheng et al., 2024] apply the Flow Matching framework to the simplex $\Delta _ { d }$ equipped with the Fisher–Rao geometry, focusing on categorical data. Similarly, Stark et al. [2024] utilize the Dirichlet distribution for discrete data generation. For

<table><tr><td>Method</td><td>Data Space</td><td>Distribution Space</td></tr><tr><td>FM</td><td> $( \mathbb { R } ^ { d } , L ^ { 2 } )$ </td><td>X</td></tr><tr><td>RFM</td><td> $( \mathcal { M } , g )$ </td><td>X</td></tr><tr><td>WFM</td><td> $( \mathbb { R } ^ { d } , L ^ { 2 } )$ </td><td> $( \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) , W )$ </td></tr><tr><td>RWEFM</td><td> $( \mathcal { M } , g )$ </td><td> $( \mathscr { P } _ { 2 } ( \mathcal { M } ) , W )$ </td></tr></table>

Table 1: Comparison of FM approaches. Only RWEFM respects both the Riemannian structure of the data space and the Wasserstein structure of the distribution space.

continuous distributions, Meta FM [Atanackovic et al., 2024] and Wasserstein FM [Haviv et al., 2024, Piening et al., 2026] learn flows directly on the Wasserstein space $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ . Crucially, these approaches are restricted to Euclidean base spaces, while our framework extends to distributions over general Riemannian manifolds.

<table><tr><td rowspan="2"></td><td colspan="6">Manifold: H2 (EMNIST)</td><td colspan="6">Manifold: S2 (MNIST)</td><td colspan="6">Manifold: T2 (KMNIST)</td></tr><tr><td colspan="2">h</td><td colspan="2">W</td><td colspan="2">y</td><td colspan="2">3</td><td colspan="2">4</td><td colspan="2">8</td><td colspan="2">ki</td><td colspan="2">na</td><td colspan="2">ma</td></tr><tr><td>Method</td><td>CD</td><td>EMD</td><td>CD</td><td>EMD</td><td>CD</td><td>EMD</td><td>CD</td><td>EMD</td><td>CD</td><td>EMD</td><td>CD</td><td>EMD</td><td>CD</td><td>EMD</td><td>CD</td><td>EMD</td><td>CD</td><td>EMD</td></tr><tr><td>PVD</td><td>0.32</td><td>0.35</td><td>0.43</td><td>0.38</td><td>0.25</td><td>0.30</td><td>0.45</td><td>0.42</td><td>0.40</td><td>0.40</td><td>0.47</td><td>0.42</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>1</td></tr><tr><td>PSF</td><td>0.18</td><td>0.33</td><td>0.06</td><td>0.30</td><td>0.21</td><td>0.31</td><td>0.19</td><td>0.35</td><td>0.17</td><td>0.29</td><td>0.23</td><td>0.39</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>FM</td><td>0.49</td><td>0.49</td><td>0.48</td><td>0.48</td><td>0.48</td><td>0.48</td><td>0.40</td><td>0.42</td><td>0.45</td><td>0.46</td><td>0.36</td><td>0.41</td><td>0.48</td><td>0.46</td><td>0.49</td><td>0.48</td><td>0.47</td><td>0.47</td></tr><tr><td>SetFM</td><td>0.37</td><td>0.35</td><td>0.32</td><td>0.33</td><td>0.34</td><td>0.33</td><td>0.26</td><td>0.27</td><td>0.29</td><td>0.29</td><td>0.41</td><td>0.35</td><td>0.28</td><td>0.28</td><td>0.24</td><td>0.24</td><td>0.28</td><td>0.29</td></tr><tr><td>WFM</td><td>0.30</td><td>0.28</td><td>0.41</td><td>0.39</td><td>0.29</td><td>0.28</td><td>0.29</td><td>0.28</td><td>0.29</td><td>0.27</td><td>0.31</td><td>0.37</td><td>0.22</td><td>0.17</td><td>0.19</td><td>0.14</td><td>0.17</td><td>0.17</td></tr><tr><td>RFM</td><td>0.48</td><td>0.48</td><td>0.44</td><td>0.45</td><td>0.47</td><td>0.48</td><td>0.30</td><td>0.38</td><td>0.42</td><td>0.45</td><td>0.33</td><td>0.38</td><td>0.47</td><td>0.46</td><td>0.49</td><td>0.48</td><td>0.47</td><td>0.47</td></tr><tr><td>SetRFM</td><td>0.30</td><td>0.31</td><td>0.34</td><td>0.33</td><td>0.31</td><td>0.31</td><td>0.38</td><td>0.38</td><td>0.38</td><td>0.38</td><td>0.39</td><td>0.40</td><td>0.26</td><td>0.28</td><td>0.23</td><td>0.23</td><td>0.26</td><td>0.28</td></tr><tr><td>RWEFM</td><td>0.15</td><td>0.09</td><td>0.23</td><td>0.10</td><td>0.16</td><td>0.11</td><td>0.18</td><td>0.18</td><td>0.19</td><td>0.18</td><td>0.16</td><td>0.16</td><td>0.17</td><td>0.18</td><td>0.16</td><td>0.16</td><td>0.15</td><td>0.18</td></tr></table>

Table 2: RWEFM on MNIST, EMNIST, and KMNIST. Benchmarking of RWEFM against FM variants and other point-cloud generative models on hyperbolic $\ ( \mathbb { H } ^ { 2 }$ , EMNIST), spherical $\left( { { \mathbb S } ^ { 2 } } , \mathrm { M N I S T } \right)$ , and toroidal $\bar { ( \mathbb { T } ^ { 2 } }$ , KMNIST) data. We report the classwise 1-NN deviation (1-NN-D) using Chamfer Distance (CD) and Earth Mover’s Distance (EMD) between generated and test point clouds. The score ranges from 0 to 0.5, where 0 corresponds to 50% accuracy in each class; lower is better. Values are averaged over 3 random seeds and 5 samplings per seed. The best three results per column are highlighted $( \mathrm { 1 s t , 2 n d , 3 r d } )$ ; ties share a rank. Dashes indicate methods not evaluated on that manifold. Full results with MMD metrics and per-manifold mean ± std are reported in Tables S3, S5 and S6 (Appendix D). Per-dataset training times for all methods are in Table S1.

## 3. Riemannian Wasserstein Entropic FM.

3.1. Flow Matching on $\mathcal { P } _ { 2 } ( \mathcal { M } )$ . RWEFM learns generative flows on $\mathcal { P } _ { 2 } ( \mathcal { M } )$ by lifting the conditional FM paradigm to $\mathcal { P } _ { 2 } ( \mathcal { P } _ { 2 } ( \mathcal { M } ) )$ ), respecting the intrinsic geometry of $\mathcal { M }$ and $\mathcal { P } _ { 2 } ( \mathcal { M } )$ . Following Section 2.2, this requires three components (C1–C3). We start with (C1), the conditions for when a vector field $V _ { t } \in T \mathscr { P } _ { 2 } ( \mathcal { M } )$ generates a probability path $\mathbb { P } _ { t } \in \mathcal { P } _ { 2 } ( \mathcal { P } _ { 2 } ( \mathcal { M } ) )$ .

3.1.1. Weak continuity equation and superposition on $\mathcal { P } _ { 2 } ( \mathcal { M } )$ (C1). The following result shows that if $( \mathbb { P } _ { t } , V _ { t } )$ solves the weak continuity equation on $\mathcal { P } _ { 2 } ( \mathcal { M } )$ then $V _ { t }$ generates $\mathbb { P } _ { t }$ in the Lagrangian sense:

Theorem 3.1. Let P be an absolutely continuous curve on $\mathcal { P } _ { 2 } ( \mathcal { P } _ { 2 } ( \mathcal { M } ) )$ , and a vector field $V _ { t } \in L ^ { 2 } ( \mathbb { P } _ { t } , T \mathcal { P } _ { 2 } ( \mathcal { M } ) )$ with $\begin{array} { r } { \int _ { \mathcal { P } _ { 2 } ( \mathcal { M } ) } \| V _ { t } ( \mu ) \| _ { L ^ { 2 } ( \mu ) } ^ { 2 } d \mathbb { P } _ { t } ( \mu ) < \infty } \end{array}$

Assume that $( \mathbb { P } _ { t } , V _ { t } )$ solves the weak continuity equation on $\mathcal { P } _ { 2 } ( \mathcal { M } )$

$$
\int _ { 0 } ^ { T } \int _ { \mathcal { P } _ { 2 } ( \mathcal { M } ) } \bigg ( \partial _ { t } \varphi _ { t } ( \mu ) + \langle \nabla _ { \mathcal { W } } \varphi _ { t } ( \mu ) , V _ { t } ( \mu ) \rangle _ { L ^ { 2 } ( \mu ) } \bigg ) d \mathbb { P } _ { t } ( \mu ) d t = 0\tag{3.1}
$$

for all smooth, compactly supported cylinder functionals $\varphi _ { t }$ . Assume moreover that $V _ { t }$ satisfies suitable regularity condition specified in Appendix C. Then, $f o r \ P _ { 0 } - a . e$ $\mu \in \mathcal P _ { 2 } ( \mathcal M )$ , there exists a unique absolutely continuous curve $\gamma _ { \mu } ( t ) : [ 0 , T ]  { \mathscr { P } _ { 2 } ( \mathcal { M } ) }$ solving

$$
\dot { \gamma } _ { \mu } ( t ) = V _ { t } ( \gamma _ { \mu } ( t ) ) \quad f o r \ a . e . \ t , \ w i t h \ \gamma _ { \mu } ( 0 ) = \mu
$$

and the map $\Phi _ { t } ( \mu ) : = \gamma _ { \mu } ( t )$ defines a flow such that $\mathbb { P } _ { t } = ( \Phi _ { t } ) _ { \# } \mathbb { P } _ { 0 }$ for all $t \in [ 0 , T ]$ The proof uses a superposition principle for random measures and is given in $\mathrm { A p - }$ pendix C.1.

3.1.2. Marginalization of the conditional vector field (C2). Let $\Pi ( \mu , \nu )$ be a probability measure on $\mathscr { P } _ { 2 } ( \mathcal { M } ) \otimes \mathscr { P } _ { 2 } ( \mathcal { M } )$ , with marginals $\begin{array} { r } { \int _ { \mu } d \Pi ( \mu , \nu ) = \mathbb { P } _ { 1 } } \end{array}$ and $\begin{array} { r } { \int _ { \nu } d \Pi ( \mu , \nu ) = \mathbb { P } _ { 0 } } \end{array}$ . Given a conditional vector field $V _ { t } ( \cdot | \mu _ { 0 } , \mu _ { 1 } )$ , conditioned on a sample $( \mu _ { 0 } , \mu _ { 1 } ) \sim \Pi$ , we define the marginal vector field $V _ { t }$ as:

$$
V _ { t } ( \mu _ { t } ) : = \mathbb { E } _ { \mathbb { P } _ { t } ( \mu , \nu | \mu _ { t } ) } [ V _ { t } ( \mu _ { t } | \mu _ { 0 } , \mu _ { 1 } ) ] = \iint _ { \mathcal { P } _ { 2 } ( M ) \times \mathcal { P } _ { 2 } ( M ) } V _ { t } ( \mu _ { t } | \mu , \nu ) d \mathbb { P } _ { t } ( \mu , \nu | \mu _ { t } )
$$

where $\begin{array} { r } { d \mathbb { P } _ { t } ( \mu , \nu | \mu _ { t } ) = \frac { d \mathbb { P } _ { t } ( \mu _ { t } | \mu , \nu ) d \Pi ( \mu , \nu ) } { d \mathbb { P } _ { t } ( \mu _ { t } ) } } \end{array}$ is the conditional probability measure. The following result establishes the weak continuity equation for the marginal pair and, under the hypotheses of Theorem 3.1, its Lagrangian interpretation.

Proposition 3.2. Assume $( \mathbb { P } _ { t } ( \cdot | \mu , \nu ) , V _ { t } ( \cdot | \mu , \nu ) )$ solves the weak continuity equation (3.1), for $\mu , \nu \Pi - a . s .$ . Then $( \mathbb { P } _ { t } , V _ { t } )$ also solves (3.1). Under the hypotheses of the superposition theorem (Theorem 3.1) for the marginal pair, V generates the marginal probability path $\mathbb { P } _ { t }$

3.1.3. RWEFM Training Objective (C3). The ideal, yet intractable, Flow Matching objective is

$$
\mathcal { L } _ { F M } = \mathbb { E } _ { t , \mu _ { t } \sim \mathbb { P } _ { t } } [ \| V _ { t } ( \mu _ { t } ) - V _ { t } ^ { \theta } ( \mu _ { t } ) \| _ { L ^ { 2 } ( \mu _ { t } ) } ^ { 2 } ] .
$$

To bypass this, we optimize the RWEFM objective:

$$
\mathcal { L } _ { R W E F M } = \mathbb { E } _ { \underset { \mu _ { t } \sim \mathbb { P } _ { t } ( \cdot | \mu , \nu ) } { t , \mu , \nu \sim \Pi , } } \left[ \left\| V _ { t } ( \mu _ { t } \ | \ \mu , \nu ) - V _ { t } ^ { \theta } ( \mu _ { t } ) \right\| _ { L ^ { 2 } ( \mu _ { t } ) } ^ { 2 } \right]\tag{3.2}
$$

As shown in Appendix C.3, $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { F M } = \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { R W E F M }$ . Combining C1–C3, we have successfully lifted flow matching to $\mathcal { P } _ { 2 } ( \mathcal { P } _ { 2 } ( \mathcal { M } ) )$

3.1.4. RWEFM interpolants. Diferent choices of conditional probability paths $\mathbb { P } _ { t } ( \cdot | \mu _ { 0 } , \mu _ { 1 } )$ are possible. In this work, we choose the McCann geodesics as the canonical interpolants between $\mu$ and $\nu \cdot$

$$
\mathbb { P } _ { t } ( \cdot | \mu , \nu ) = \delta _ { \mu _ { t } } \mathrm { ~ w i t h ~ } \mu _ { t } = ( \psi _ { t } ) _ { \# } \mu \mathrm { ~ a n d ~ } \psi _ { t } ( x ) = \exp _ { x } ( t \log _ { x } ( T ^ { \mu \to \nu } ( x ) ) ) .\tag{3.3}
$$

For such conditional probability path, the conditional vector field that solves (3.1) is given by:

$$
V _ { t } ( \cdot | \mu , \nu ) = \frac { 1 } { 1 - t } \log _ { x _ { t } } T ^ { \mu \to \nu } ( \psi _ { t } ^ { - 1 } ( x _ { t } ) ) .\tag{3.4}
$$

We verify that $V _ { t } ( \cdot | \mu , \nu ) \in T _ { \mu _ { t } } \mathscr { P } _ { 2 } ( \mathcal { M } )$ with $\log _ { x _ { t } } T ^ { \mu \to \nu } ( \psi _ { t } ^ { - 1 } ( x _ { t } ) ) = \log _ { x _ { t } } T ^ { \mu \to \nu } ( x _ { t } ) =$ $- \nabla \varphi ( x _ { t } )$ for some $\varphi \in C _ { c } ^ { \infty } ( \mathcal { M } )$

3.2. The Riemannian Entropic Map. A crucial step of computing the RWEFM objective is obtaining the Monge map $T ^ { \mu \to \nu }$ between two distributions. To this end, we propose the Riemannian Entropic Map $T _ { \varepsilon } ( x )$ , a scalable estimator for $T ^ { \mu \to \nu }$ on Riemannian manifolds based on finite samples from $\mu$ and $\nu$ and fast entropic OT solvers.

Like the Euclidean entropic map [Pooladian and Niles-Weed, 2021], our estimator leverages the entropic optimal coupling $\pi _ { \varepsilon }$ obtained from the Sinkhorn algorithm, computed using the Riemannian distance cost matrix $C _ { i j } = { \textstyle \frac { 1 } { 2 } } d _ { g } ^ { 2 } ( x _ { i } , y _ { j } )$ between samples $\{ x _ { i } \} _ { i = 1 } ^ { N } \sim \mu$ and $\{ y _ { j } \} _ { j = 1 } ^ { N } \sim \nu$

Definition 3.3 (Riemannian Entropic Map). Let $\pi _ { \varepsilon }$ be the optimal entropic coupling. We define the Riemannian entropic map $T _ { \varepsilon } : \mathcal { M }  \mathcal { M }$ as:

$$
T _ { \varepsilon } ( x ) : = \exp _ { x } \left( \int _ { \mathcal { M } } \log _ { x } ( y ) d \pi _ { \varepsilon } ( y | x ) \right) .\tag{3.5}
$$

Theorem 3.4. Let $f _ { \varepsilon }$ be the optimal dual potential of the entropic optimal transport problem $( E q \ ( 2 . 2 ) )$ . Under primal feasibility conditions, the Riemannian entropic map arises as the gradient of the dual potential: $T _ { \varepsilon } ( x ) = \exp _ { x } { \bigl ( } - \nabla f _ { \varepsilon } ( x ) { \bigr ) }$ The proof is in Appendix A; when $\mathcal { M } = \mathbb { R } ^ { d }$ , Eq (3.5) recovers the Euclidean map $T _ { \varepsilon } = \mathbb { E } _ { \pi _ { \varepsilon } } [ Y | X = x ]$

Computationally, this definition implies a straightforward lift-average-retract procedure for empirical measures. To evaluate the map at a source point x: (i) lift by computing the tangent vectors $v _ { j } = \log _ { x } ( y _ { j } )$ for all target points $y _ { j }$ , mapping the geometry from M to the vector space $T _ { x } { \mathcal { M } } ; ( { \mathrm { i i } } )$ average to calculate the weighted mean $\begin{array} { r } { \bar { v } = \sum _ { i } \pi _ { \varepsilon } ( y _ { j } | x ) v _ { j } ; } \end{array}$ (iii) retract by applying the exponential map to project back to the manifold: $T _ { \varepsilon } ( x ) = \exp _ { x } ( \bar { v } )$ . We highlight that this procedure only requires access to the Riemannian exponential and logarithm maps, making it applicable to a wide range of manifolds. When these maps are not available, our estimator can still be eficiently computed with only access to the distance function $d _ { g } ( x , y )$ , as shown in Appendix $\mathrm { A . 6 } ;$ we exploit this to apply RWEFM on a triangulated mesh with no closed-form geometry (Section 4.2.3). Furthermore, the estimator has the same complexity as the Euclidean entropic map, making it equally scalable to large datasets.

To justify using $T _ { \varepsilon }$ as a reliable proxy for the true Monge map $T _ { 0 }$ in our learning objective, we establish the following statistical bound. This result ensures that the bias introduced by entropic regularization and finite sampling is controlled.

Theorem 3.5 (Statistical performance). Let $\widehat { T } _ { \varepsilon , ( n , n ) }$ be the canonical out-ofsample entropic map computed from n iid observations from each of µ and $\nu ,$ with the two samples independent. Under Assumptions $A . 4 , \ A . 5 ,$ and A.6, there exist constants $C < \infty$ and $\varepsilon _ { 0 } > 0$ such that, for $n \geq 2$ and $0 < \varepsilon \le \varepsilon _ { 0 }$

$$
\mathbb { E } \int _ { \Omega } d _ { g } ^ { 2 } \Big ( \widehat { T } _ { \varepsilon , ( n , n ) } ( x ) , T _ { 0 } ( x ) \Big ) \ d \mu ( x ) \leq C \left[ \varepsilon \log \Big ( \frac { e } { \varepsilon } \Big ) + \varepsilon ^ { - s _ { d } } \frac { \log ( n + 1 ) } { \sqrt { n } } \right] ,
$$

where $s _ { d } = \operatorname* { m a x } \{ 1 , d / 2 \}$ and $d = \dim { \mathcal { M } }$ . The constants depend on the geometry and regularity assumptions. The proof is given in Appendix A.7.

![](images/71086931d3e095182d86c7af9857e833ddd921720b876d0b95bb17764aee4e0b.jpg)  
Fig. 3: Riemannian Entropic Map. Gaussian-to-checkerboard transport on $\mathbb { S } ^ { 2 }$ using Euclidean (left) and Riemannian (middle) entropic maps. The Riemannian approach respects the underlying geometry, resulting in a more accurate mapping (see Figure S2).

Algorithm 3.1 RWEFM Training Step   
Input: $\mathbb { P } _ { 0 } , \mathbb { P } _ { 1 }$   
Sample $\mu \sim \mathbb { P } _ { 0 } , \nu \sim \mathbb { P } _ { 1 }$   
Sample $\{ x _ { i } \} _ { i = 1 } ^ { N } \sim \mu , \{ y _ { j } \} _ { j = 1 } ^ { M } \sim \nu$   
Compute π via Sinkhorn $( C , \varepsilon )$ , Sample $t \sim \mathcal { U } [ 0 , 1 ]$   
for each $x _ { i }$ do   
$w _ { i j } = { \pi } _ { i j } / \sum _ { k } { \pi } _ { i k }$   
$\begin{array} { r } { T _ { \varepsilon } ( x _ { i } ) = \exp _ { x _ { i } } \left( \sum _ { j } w _ { i j } \log _ { x _ { i } } ( y _ { j } ) \right) } \end{array}$   
$z _ { i } = \mathrm { e x p } _ { x _ { i } } ( t \log _ { x _ { i } } ( T _ { \varepsilon } ( x _ { i } ) ) )$   
$v _ { i } = \log _ { z _ { i } } \dot { ( } T _ { \varepsilon } ( x _ { i } ) \dot { ) } / ( 1 - t )$   
$\hat { v } _ { i } = v _ { \theta } ( \dot { z _ { i } } , t , \{ z _ { i } \} _ { i = 1 } ^ { N } )$   
$\begin{array} { r } { \mathcal { L } = \frac { 1 } { N } \sum _ { i } \| \hat { v } _ { i } - v _ { i } \| _ { g ( z _ { i } ) } ^ { 2 } } \end{array}$   
Update $\theta \gets \theta - \eta \nabla _ { \theta } \mathcal { L }$

Algorithm 3.2 RWEFM Generation   
Input: $\mu \sim \mathbb { P } _ { 0 } ,$ discretization step dt   
Sample $X _ { 0 } = \{ x _ { i } \} _ { i = 1 } ^ { N } \sim \mu$   
for $t \gets 0$ to 1 − dt step dt do   
Compute $v _ { i } = v _ { \theta } ( x _ { i } , t , \{ x _ { i } \} _ { i = 1 } ^ { N } )$   
for each x<sub>i</sub> do   
$x _ { i } \gets \exp _ { x _ { i } } ( v _ { i } \cdot d t )$   
Return: $X _ { 1 } = \{ x _ { i } \} _ { i = 1 } ^ { N }$

3.3. Algorithm and Implementation. Here we detail how RWEFM is implemented in practice. The training algorithm is presented in Algorithm 3.1. At each training step, we sample distributions $\mu$ and $\nu$ from the meta-distributions $\mathbb { P } _ { 0 }$ and $\mathbb { P } _ { 1 }$ , represented as point clouds $X _ { 0 } = \{ x _ { i } \} _ { i = 1 } ^ { N } , x _ { i } \sim \mu$ and $X _ { 1 } = \{ y _ { j } \} _ { j = 1 } ^ { M } , y _ { j } \sim \nu .$ . We compute the Riemannian entropic map (Eq (3.5)) to generate an interpolating point cloud $X _ { t } = \{ z _ { i } \} _ { i = 1 } ^ { N }$ with $z _ { i } = \exp _ { x _ { i } } ( t \log _ { x _ { i } } ( T _ { \varepsilon } ( x _ { i } ) ) )$ (Eq (3.3)) and target velocity vectors $\begin{array} { r } { V _ { t } = \{ \frac { 1 } { 1 - t } \log _ { z _ { i } } ( T _ { \varepsilon } ( x _ { i } ) ) \} _ { i = 1 } ^ { N } \ \mathrm { \backslash } \mathrm { ( E q \ ( 3 . 4 ) ) } } \end{array}$ . Since the data is set-structured, we parameterize the neural network $v _ { \theta } ( X _ { t } , t ) : \mathcal { M } ^ { N } \to T _ { X _ { t } } \mathcal { M } ^ { N }$ using self-attention architectures, mapping point sets to tangent vectors. The loss is defined as the mean squared geodesic norm between the network prediction $\hat { v } _ { i }$ and the target $v _ { i } ~ \left( \mathrm { E q } ~ ( 3 . 2 ) \right)$ . For generation (Algorithm 3.2), we start from $X _ { 0 } \sim \mu \sim \mathbb { P } _ { 0 }$ and integrate $v _ { \theta }$ over time to obtain the final sample $X _ { 1 }$

Appendix D details hyperparameters and architecture. RWEFM typically trains in a few hours on a single GPU. For large datasets like the single-cell atlas, we trained with 1024 cells sampled from each distribution, still achieving high-quality results in less than 24 hours of training. The size of the point clouds and the entropic regularization $\varepsilon$ are the main hyper-parameters impacting computational complexity, with per-dataset training times for all methods reported in Table S1. Training time can be meaningfully reduced by increasing ε or decreasing the number of sampled particles, at a marginal cost to generation quality, as studied in detail in Section D.9.

![](images/981d9edddce4f699bcb929054dc7665185a1a0013c789d01773d192ceeef11dc.jpg)  
Fig. 4: Generative Modeling of Single-Cell Lung Samples. UMAP visualization of real and generated scRNA-seq samples from healthy and cancer lung tissues. The generated samples demonstrate high congruence with the real distributions. Furthermore, operating in the foundation model’s latent space enables direct analysis of generated data, such as accurate cell typing, highlighting the utility of generating in the (non-Euclidean) latent spaces of foundation models.

## 4. Results.

4.1. The Riemannian Entropic Map is a faithful estimator of the Monge map on M. We begin by qualitatively demonstrating the efectiveness of the Riemannian entropic map. Figure 3 illustrates the transport error on the sphere $\mathbb { S } ^ { 2 }$ , comparing our Riemannian estimator against a Euclidean baseline. By strictly adhering to the underlying geometry, the Riemannian entropic map yields a significantly more accurate coupling, whereas the Euclidean approach incurs high distortion as it ignores the manifold curvature. We provide quantitative evaluation of its performance on constructed examples on the sphere $\mathbb { S } ^ { 2 }$ and hyperbolic space $\mathbb { H } ^ { 2 }$ in Appendix D.8. The ground truth OT map is generated by taking the exponential map of the gradient of a distance function between a point $x \in \mathcal { M }$ and a fixed attractor $\bar { x } \in \mathcal { M }$ . Our results confirm that our estimator is as accurate as the unregularized OT map (Eq (2.1)), albeit much faster, and outperforms the Euclidean entropic map, which neglects the underlying geometry (Figure S2).

## 4.2. Benchmarking RWEFM on synthetic and real-world scientific datasets.

4.2.1. Baselines and Evaluation Metrics. We benchmark RWEFM against several baselines. FM [Lipman et al., 2022] and RFM [Chen and Lipman, 2023] operate on individual data points. WFM [Haviv et al., 2024, Atanackovic et al., 2024] learns flows on the Wasserstein space but is restricted to Euclidean domains. We also introduce SetFM/SetRFM, which apply the FM objective to point clouds without OT couplings. For point cloud data, we include PVD [Zhou et al., 2021] and PSF [Wu et al., 2023], trained in Euclidean space and projected onto the manifolds. All set-level methods share the same self-attention architecture, FM/RFM use an MLP and we project generated samples from non-Riemannian methods onto the manifold for fair comparison. We evaluate model performance using classwise 1-NN deviation (1-NN-D) and MMD, with Chamfer Distance (CD) and Earth Mover’s Distance (EMD) between point clouds (Appendix D.3). For equally sized sets of real and generated point clouds, let $a _ { \mathrm { r e a l } }$ and $a _ { \mathrm { g e n } }$ be their respective classwise nearest-neighbor classification accuracies. We report $\mathrm { \mathop { 1 - N N - D } } : = \frac { 1 } { 2 } \left( \left| a _ { \mathrm { r e a l } } - \frac { 1 } { 2 } \right| + \left| a _ { \mathrm { g e n } } - \frac { 1 } { 2 } \right| \right)$ This score lies in [0, 0.5] and is minimized when both classwise accuracies equal 0.5. Taking the absolute deviations before averaging prevents opposite classwise deviations from canceling.

4.2.2. MNIST, EMNIST, and KMNIST on the Sphere, Hyperbolic Space, and Torus. We evaluate RWEFM on MNIST [LeCun et al., 1998], EMNIST [Cohen et al., 2017], and KMNIST [ROIS-CODH, 2018] datasets mapped onto Riemannian manifolds: MNIST digits on the sphere $\mathbb { S } ^ { 2 }$ , EMNIST letters on hyperbolic space $\mathbb { H } ^ { 2 }$ , and KMNIST characters on the torus $\mathbb { T } ^ { 2 }$ . RWEFM learns high-quality flows and outperforms baselines in all settings (Figure 2, Tables 2,S3). Notably, on the torus $\mathbb { T } ^ { 2 }$ , WFM and RWEFM achieve comparable performance. This is expected as $\mathbb { T } ^ { 2 }$ is a flat manifold with zero curvature and the geometric advantage of RWEFM’s Riemannian OT over WFM’s Euclidean OT diminishes accordingly. Unsurprisingly, FM/RFM perform worst as they are unable to capture the dependencies between individual points required to generate meaningful point clouds.

4.2.3. Beyond closed-form manifolds: distributions on a general triangulated mesh. The manifolds above admit closed-form geodesics, exponential and logarithm maps. Many scientific geometries—triangulated surfaces, learned metrics, implicit surfaces—do not.

RWEFM extends naturally to this setting: both the McCann interpolant and the Riemannian entropic map (Section 3.2) can be evaluated from only a geodesic distance $d _ { g }$ and a projection operator onto the manifold, without analytic exp/log maps (Appendix A.6).

As a proof of concept, we generate distributions on the surface of the Stanford bunny, a triangulated mesh with no analytic geometry, using MNIST digits as the empirical distributions. We endow the mesh with the spectral (biharmonic) premetric of Chen and Lipman [2023], computed from its smallest Laplace–Beltrami eigenpairs, and lay each digit onto a fixed tangent chart on the bunny’s flank (Appendix D.5). Figure 5 shows point clouds generated by RWEFM for three digit classes and the samples lie on the curved surface and recover the digit morphology.

Quantitatively (Table 3), the meshaware methods (RWEFM, SetRFM) sharply outperform their Euclidean counterparts (WFM, SetFM), which ignore the surface and generally attain or approach the maximum classwise 1-NN deviation $( D _ { \mathrm { 1 N N } } = 0 . 5 )$ , indicating large departures from 50% classwise accuracy. Respecting the intrinsic geometry therefore remains decisive even when the manifold has no closed-form description, demonstrating that RWEFM applies to arbitrary geometries.

4.2.4. Single-cell RNA-seq samples on Riemannian spaces. In singlecell genomics, early generative models focused on synthesizing individual cells. A growing body of work now targets a fundamentally harder task: generating whole samples—each sample being an empirical

<table><tr><td rowspan="2"></td><td colspan="6">Manifold:  $\mathcal { M } _ { \mathrm { m e s h } }$ </td></tr><tr><td colspan="2">Digit 0</td><td colspan="2">Digit 2</td><td colspan="2">Digit 9</td></tr><tr><td>Method</td><td>CD</td><td>EMD</td><td>CD</td><td>EMD</td><td>CD</td><td>EMD</td></tr><tr><td>SetFM</td><td>0.50</td><td>0.50</td><td>0.50</td><td>0.50</td><td>0.50</td><td>0.50</td></tr><tr><td>WFM</td><td>0.50</td><td>0.50</td><td>0.40</td><td>0.42</td><td>0.50</td><td>0.50</td></tr><tr><td>SetRFM</td><td>0.30</td><td>0.26</td><td>0.29</td><td>0.23</td><td>0.37</td><td>0.24</td></tr><tr><td>RWEFM</td><td>0.30</td><td>0.28</td><td>0.26</td><td>0.21</td><td>0.29</td><td>0.21</td></tr></table>

Table 3: RWEFM on a general triangulated mesh (Stanford bunny). Here $\mathcal { M } _ { \mathrm { m e s h } }$ denotes the Stanford bunny surface represented by a triangulated mesh. Classwise 1-NN deviation (1-NN-D; lower is better) under the mesh spectral metric for MNIST digits generated on the bunny (best per column bold; 3 seeds $\times \ 5$ samplings). RWEFM and WFM use the sampled map (Appendix D.5); MMD in Table S2.

distribution of thousands of cells representing a patient or tissue. This sample-level perspective is essential for capturing inter-sample heterogeneity in disease modeling and perturbation studies [Boyeau et al., 2025, Boiarsky et al., 2025, Haviv et al., 2024,

![](images/b2c4231e07f313959b430711a77b103e47eb59a8911d3637dd6c96f23c612531.jpg)  
Fig. 5: Distributions generated on the Stanford bunny. RWEFM-generated empirical distributions on a general triangulated mesh with no closed-form geometry, showing two samples per digit class (0, 2, 9). Generated points lie on the curved surface and recover the digit morphology, illustrating RWEFM on arbitrary geometries (Table 3).

Atanackovic et al., 2024, Klein et al., 2025]. Concretely, each training example is a point cloud $ { \mu } = \{ x _ { i } \} _ { i = 1 } ^ { N }$ of cell embeddings for one patient sample, and the goal is to generate new point clouds that faithfully reproduce the distribution of real samples.
<table><tr><td rowspan="2">Method</td><td colspan="2">Manifold:  $\overline { { \mathbb { S } ^ { 1 2 8 - 1 } } }$ </td><td colspan="2">Manifold:</td><td> $\mathbb { T } ^ { 2 }$ </td></tr><tr><td>1-NN-D (CD)</td><td>MMD (EMD)</td><td> $W _ { 2 }$ </td><td> $W _ { 1 }$ </td><td>MMD</td></tr><tr><td>SetFM</td><td>0.4460</td><td>0.1026</td><td>0.4472</td><td>0.4097</td><td>0.0114</td></tr><tr><td>WFM</td><td>0.2945</td><td>0.0324</td><td>0.4059</td><td>0.4023</td><td>0.0120</td></tr><tr><td>SetRFM</td><td>0.3754</td><td>0.0300</td><td>0.4608</td><td>0.4126</td><td>0.0119</td></tr><tr><td>RWEFM</td><td>0.1889</td><td>0.0140</td><td>0.3970</td><td>0.3850</td><td>0.0097</td></tr></table>

Table 4: Generation of cells and torsion angles on manifolds. (Left) whole single-cell RNAseq samples on $\mathbb { S } ^ { 1 2 8 - 1 }$ (SCimilarity embeddings); (right) per-protein torsion-angle distributions on $\mathbb { T } ^ { 2 }$ (ESM-conditioned). Extended metrics in Table S4; times in Table S1.

A further challenge is that cell embeddings from foundation models such as SCimilarity [Heimberg et al., 2025] naturally live on non-Euclidean manifolds such as hyperspheres $( \mathbb { S } ^ { d - 1 } )$ or hyperbolic spaces [Ding and Regev, 2021], so naive Euclidean generation distorts the learned geometry. RWEFM directly addresses both challenges: it generates distributions (whole samples) on the manifold by learning a flow on the Wasserstein space $\mathcal { P } _ { 2 } ( \mathbb { S } ^ { 1 2 8 - 1 } )$ . We evaluate on generating whole single-cell blood samples in SCimilarity’s hyperspherical embedding space and find that RWEFM substantially outperforms Euclidean and non-distributional baselines (Table 4, Figures 4,S1). The generated samples accurately recapitulate real cell-type compositions and enable direct cell-typing analysis, demonstrating the utility of respecting the geometry of foundation model latent spaces.

4.2.5. Protein torsion angle distributions on the torus. Protein backbone conformations can be characterized by dihedral angles $\phi$ and $\psi .$ . The joint angular distribution is depicted using the Ramachandran plot, which reveals energetically favorable conformations and structural motifs such as α-helices and β-sheets. Each protein is thus characterized by a distribution of torsion angles. Since both angles are periodic, this distribution lives on the flat torus $\mathbb { T } ^ { 2 }$ , a common setting for Riemannian generative models [Chen and Lipman, 2023, Davis et al., 2025].

We apply RWEFM to generate per-protein torsion angle distributions from md-CATH [Mirarchi et al., 2024]. For each protein, we aggregate the $( \phi , \psi )$ angles across all residues during MD simulation, producing an empirical distribution over $\mathbb { T } ^ { 2 }$ . Unlike prior work that generates individual angle pairs, we generate entire distributions conditioned on ESM embeddings [Hayes et al., 2025] of the protein sequence. On unseen test-set proteins, we find that RWEFM accurately captures the complex, multimodal structure of torsion angle distributions (Figure 6). Quantitatively, RWEFM outperforms Euclidean and non-distributional baselines (Table 4), demonstrating the benefits of integrating Riemannian geometry with flow matching on $\mathcal { P } _ { 2 } ( \mathcal { M } )$ for modeling protein conformational landscapes.

![](images/5c510db9ed484f72446a4fdac13e89d7032560723131551b5adb3464c765daf7.jpg)  
Fig. 6: Torsion angle generation on the torus $\mathbb { T } ^ { 2 }$ . True and generated distributions of protein backbone torsion angles $( \phi , \psi )$ for held-out mdCATH proteins, conditioned on ESM embeddings of their sequences (first 20 amino acids shown in subplot titles). RWEFM accurately captures the multimodal structure of conformational distributions.

5. Discussion. We have introduced Riemannian Wasserstein Entropic Flow Matching (RWEFM), a novel framework for generative modeling of distributions on Riemannian manifolds. We showed that our framework is theoretically sound, computationally tractable, and empirically outperforms previous approaches that either fail to capture the distributional or geometric aspects of the data. Our proposed Riemannian entropic map is an essential part of our approach, providing highly scalable and accurate estimation of the OT map. Our empirical benchmark spans multiple scientific applications, including single-cell genomics and protein conformation, highlighting the diversity and abundance of scientific problems that could benefit from explicitly incorporating the geometry and distributional nature of the data.

References.

John T Abatzoglou, Solomon Z Dobrowski, Sean A Parks, and Katherine C Hegewisch. Terraclimate, a high-resolution global dataset of monthly climate and climatic water balance from 1958–2015. Scientific data, 5(1):170191, 2018.

Luigi Ambrosio, Elia Brué, and Daniele Semola. Lectures on optimal transport, volume 130. Springer, 2021.

Lazar Atanackovic, Xi Zhang, Brandon Amos, Mathieu Blanchette, Leo J Lee, Yoshua Bengio, Alexander Tong, and Kirill Neklyudov. Meta flow matching: Integrating vector fields on the wasserstein manifold. arXiv preprint arXiv:2408.14608, 2024.

Simon Axelrod and Rafael Gomez-Bombarelli. Geom, energy-annotated molecular conformations for property prediction and molecular generation. Scientific Data, 9 (1):185, 2022.

Rebecca Boiarsky, Johann Wenckstern, Nicholas J Haradhvala, Gad Getz, and David Sontag. A difusion-based autoencoder for learning patient-level representations from single-cell data. bioRxiv, 2025.

Clément Bonet, Christophe Vauthier, and Anna Korba. Flowing datasets with wasserstein over wasserstein gradient flows. arXiv preprint arXiv:2506.07534, 2025.

Avishek Joey Bose, Tara Akhound-Sadegh, Guillaume Huguet, Kilian Fatras, Jarrid Rector-Brooks, Cheng-Hao Liu, Andrei Cristian Nica, Maksym Korablyov, Michael Bronstein, and Alexander Tong. Se (3)-stochastic flow matching for protein backbone generation. arXiv preprint arXiv:2310.02391, 2023.

Pierre Boyeau, Justin Hong, Adam Gayoso, Martin Kim, José L McFaline-Figueroa, Michael I Jordan, Elham Azizi, Can Ergen, and Nir Yosef. Deep generative modeling of sample-level heterogeneity in single-cell genomics. Nature Methods, 22(11):2264– 2274, 2025.

Emmanuel Candes, Yingying Fan, Lucas Janson, and Jinchi Lv. Panning for gold:‘model-x’knockofs for high dimensional controlled variable selection. Journal of the Royal Statistical Society Series B: Statistical Methodology, 80(3):551–577, 2018.

Ricky TQ Chen and Yaron Lipman. Flow matching on general geometries. arXiv preprint arXiv:2302.03660, 2023.

Chaoran Cheng, Jiahan Li, Jian Peng, and Ge Liu. Categorical flow matching on statistical manifolds. arXiv preprint arXiv:2405.16441, 2024.

Gregory Cohen, Saeed Afshar, Jonathan Tapson, and Andre Van Schaik. Emnist: Extending mnist to handwritten letters. In 2017 international joint conference on neural networks (IJCNN), pages 2921–2926. IEEE, 2017.

Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 26, 2013.

Marco Cuturi, Laetitia Meng-Papaxanthos, Yingtao Tian, Charlotte Bunne, Geof Davis, and Olivier Teboul. Optimal transport tools (ott): A jax toolbox for all things wasserstein. arXiv preprint arXiv:2201.12324, 2022.

CZI Cell Science Program, Shibla Abdulla, Brian Aevermann, Pedro Assis, Seve Badajoz, Sidney M Bell, Emanuele Bezzi, Batuhan Cakir, Jim Chafer, Signe Chambers, et al. Cz cellxgene discover: a single-cell data platform for scalable exploration, analysis and modeling of aggregated data. Nucleic acids research, 53 (D1):D886–D900, 2025.

Oscar Davis, Samuel Kessler, Mircea Petrache, Ismail I Ceylan, Michael Bronstein, and Avishek J Bose. Fisher flow matching for generative modeling over discrete data. Advances in Neural Information Processing Systems, 37:139054–139084, 2024.

Oscar Davis, Michael S Albergo, Nicholas M Bofi, Michael M Bronstein, and Avishek Joey Bose. Generalised flow maps for few-step generative modelling on

riemannian manifolds. arXiv preprint arXiv:2510.21608, 2025.

Valentin De Bortoli, Emile Mathieu, Michael Hutchinson, James Thornton, Yee Whye Teh, and Arnaud Doucet. Riemannian score-based generative modelling. Advances in neural information processing systems, 35:2406–2422, 2022.

Jiarui Ding and Aviv Regev. Deep generative model embedding of single-cell rna-seq profiles on hyperspheres and hyperbolic spaces. Nature communications, 12(1):2554, 2021.

Pedram Emami and Brendan Pass. Optimal transport with optimal transport cost: the monge–kantorovich problem on wasserstein spaces. Calculus of Variations and Partial Diferential Equations, 64(2):43, 2025.

Roy Frostig, Matthew James Johnson, and Chris Leary. Compiling machine learning programs via high-level tracing. In SysML conference 2018, 2019.

Nate Gruver, Samuel Stanton, Nathan Frey, Tim GJ Rudner, Isidro Hotzel, Julien Lafrance-Vanasse, Arvind Rajpal, Kyunghyun Cho, and Andrew G Wilson. Protein design with guided discrete difusion. Advances in neural information processing systems, 36:12489–12517, 2023.

Doron Haviv, Aram-Alexandre Pooladian, Dana Pe’er, and Brandon Amos. Wasserstein flow matching: Generative modeling over families of distributions. arXiv preprint arXiv:2411.00698, 2024.

Thomas Hayes, Roshan Rao, Halil Akin, Nicholas J Sofroniew, Deniz Oktay, Zeming Lin, Robert Verkuil, Vincent Q Tran, Jonathan Deaton, Marius Wiggert, et al. Simulating 500 million years of evolution with a language model. Science, 387(6736): 850–858, 2025.

Graham Heimberg, Tony Kuo, Daryle J DePianto, Omar Salem, Tobias Heigl, Nathaniel Diamant, Gabriele Scalia, Tommaso Biancalani, Shannon J Turley, Jason R Rock, et al. A cell atlas foundation model for scalable search of similar human cells. Nature, 638(8052):1085–1094, 2025.

Jian Huang, Yuling Jiao, Zhen Li, Shiao Liu, Yang Wang, and Yunfei Yang. An error analysis of generative adversarial networks for learning distributions. Journal of machine learning research, 23(116):1–43, 2022.

Guillaume Huguet, Alexander Tong, Edward De Brouwer, Yanlei Zhang, Guy Wolf, Ian Adelstein, and Smita Krishnaswamy. A heat difusion perspective on geodesic preserving dimensionality reduction. Advances in Neural Information Processing Systems, 36:6986–7016, 2023.

Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for eficient video generative modeling. In International Conference on Learning Representations, volume 2025, pages 23378–23402, 2025.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Dominik Klein, Jonas Simon Fleck, Daniil Bobrovskiy, Lea Zimmermann, Sören Becker, Alessandro Palma, Leander Dony, Alejandro Tejada-Lapuerta, Guillaume Huguet, Hsiu-Chuan Lin, et al. Cellflow enables generative single-cell phenotype modeling with flow matching. bioRxiv, 2025.

Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Hafner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278– 2324, 1998.

Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial difusion distillation. arXiv preprint arXiv:2402.13929, 2024.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le.

Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky TQ Chen, David Lopez-Paz, Heli Ben-Hamu, and Itai Gat. Flow matching guide and code. arXiv preprint arXiv:2412.06264, 2024.

Robert J McCann. Polar factorization of maps on riemannian manifolds. Geometric & Functional Analysis GAFA, 11(3):589–608, 2001.

Antonio Mirarchi, Toni Giorgino, and Gianni De Fabritiis. mdcath: A large-scale md dataset for data-driven computational biophysics. Scientific Data, 11(1):1299, 2024.

Moritz Piening, Richard Duong, and Gabriele Steidl. Generalized wasserstein flow matching: Transport plans, everywhere, all at once. arXiv preprint arXiv:2605.08424, 2026.

Alessandro Pinzi. A study of the metric measure space of probability measures via a purely atomic superposition principle. Nonlinear Analysis, 273:114207, 2026.

Alessandro Pinzi and Giuseppe Savaré. Nested superposition principle for random measures and the geometry of the wasserstein on wasserstein space. arXiv preprint arXiv:2510.07523, 2025.

Aram-Alexandre Pooladian and Jonathan Niles-Weed. Entropic estimation of optimal transport maps. arXiv preprint arXiv:2109.12004, 2021.

Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, and Ricky TQ Chen. Multisample flow matching: Straightening flows with minibatch couplings. arXiv preprint arXiv:2304.14772, 2023.

ROIS-CODH. KMNIST dataset. https://github.com/rois-codh/kmnist, 2018.

Arne Schneuing, Charles Harris, Yuanqi Du, Kieran Didi, Arian Jamasb, Ilia Igashov, Weitao Du, Carla Gomes, Tom L Blundell, Pietro Lio, et al. Structure-based drug design with equivariant difusion models. Nature Computational Science, 4(12): 899–909, 2024.

Hannes Stark, Bowen Jing, Chenyu Wang, Gabriele Corso, Bonnie Berger, Regina Barzilay, and Tommi Jaakkola. Dirichlet flow matching with applications to DNA sequence design. arXiv preprint arXiv:2402.05841, 2024.

Alexander Tong, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Kilian Fatras, Guy Wolf, and Yoshua Bengio. Conditional flow matching: Simulation-free dynamic optimal transport. arXiv preprint arXiv:2302.00482, 2023.

Cédric Villani. Optimal transport: old and new, volume 338. Springer, 2008.

Lemeng Wu, Dilin Wang, Chengyue Gong, Xingchao Liu, Yunyang Xiong, Rakesh Ranjan, Raghuraman Krishnamoorthi, Vikas Chandra, and Qiang Liu. Fast point cloud generation with straight flows. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9445–9454, 2023.

Kisung You. Barycentric projections of optimal transport plans on riemannian manifolds. arXiv preprint arXiv:2606.07926, 2026.

Linqi Zhou, Yilun Du, and Jiajun Wu. 3d shape generation and completion through point-voxel difusion. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5826–5835, 2021.

Zhenyu Zhu, Francesco Locatello, and Volkan Cevher. Sample complexity bounds for score-matching: Causal discovery and generative modeling. Advances in Neural Information Processing Systems, 36:3325–3337, 2023.

## Appendix contents

Appendix A. Entropic Optimal Transport in Riemannian spaces 18   
A.1 Problem Formulation . 18   
A.2 The Riemannian Entropic Map 19   
A.3 Properties of the Estimator 20   
A.4 McCann Interpolation via the Entropic Map 21   
A.5 Out-of-Sample Estimation 21   
A.6 Computing the Riemannian entropic map without analytic exp and   
log maps . 22   
A.7 Statistical performance of the estimator 23   
Appendix B. A short introduction to flow matching . . . . . 31   
Appendix C. Lifting Flow Matching to P (M) 33   
C.1 Continuity equation and Lagrangian flows on P<sub>2</sub>(M) 33   
C.2 Marginal vector field generates marginal probability path . 39   
C.3 Equivalence of conditional and non-conditional flow matching losses 41   
Appendix D. Experimental Details and additional results 42   
D.1 Neural Network Architecture & Training 42   
D.2 Geometric Operations on Manifolds . 44   
D.3 Benchmarking metrics for generation of distributions on Manifolds 45   
D.4 MNIST & EMNIST on Sphere and Hyperbolic Space 48   
D.5 Distributions on a General Mesh (Stanford Bunny) 48   
D.6 de novo generation of Single-Cell Samples on Spherical spaces 49   
D.7 Protein Torsion Angle Generation on Torus 50   
D.8 Benchmarking of Riemannian Entropic Map 50   
D.9 Training Time versus Generation Quality Tradeof . 51   
D.10 Additional Metrics and Error Values 52

Appendix A. Entropic Optimal Transport in Riemannian spaces.

Recovering the optimal transport map $T _ { 0 }$ between a source distribution $\mu$ and a target distribution $\nu$ is a central task in applied optimal transport. However, closed-form solutions for $T _ { 0 }$ are exceedingly rare, typically existing only for univariate distributions $\mathrm { o r }$ Gaussians in Euclidean space. Consequently, in the general setting of Riemannian manifolds, we must resort to statistical estimation from finite samples.

To estimate the map eficiently, we leverage Entropic Optimal Transport (EOT). Unlike unregularized OT, which requires cubic-time combinatorial solvers $\left( \mathrm { e . g . } \right.$ , the Hungarian algorithm), EOT can be solved using Sinkhorn’s algorithm. On a Riemannian manifold, this simply requires computing the geodesic distance matrix between samples. Once the entropic coupling is obtained, we extend the estimator proposed by Pooladian and Niles-Weed [2021] to the Riemannian setting.

A.1. Problem Formulation. Let $( \mathcal { M } , g )$ be a complete Riemannian manifold without boundary, and let $d ( x , y )$ denote the geodesic distance. The classical Monge problem seeks a map $T$ minimizing the transport cost:

$$
\operatorname* { i n f } _ { T \in \mathcal { T } ( \mu , \nu ) } \int _ { \mathcal { M } } \frac { 1 } { 2 } d ^ { 2 } ( x , T ( x ) ) d \mu ( x ) .
$$

The Brenier-McCann theorem shows that this problem is equivalent to

$$
\operatorname* { s u p } _ { \phi \in L ^ { 1 } ( P ) } \int _ { \mathcal { M } } \phi ( x ) d \mu ( x ) + \int _ { \mathcal { M } } \phi ^ { c } ( x ) d \nu ( x ) .\tag{A.1}
$$

where the c-transform is defined as

$$
\phi ^ { c } ( y ) = \operatorname * { i n f } _ { x \in \mathcal { M } } \{ \frac { 1 } { 2 } d ^ { 2 } ( x , y ) - \phi ( x ) \} .
$$

The optimal transport map is then given by $T _ { 0 } ( x ) = \exp _ { x } ( - \nabla \phi _ { 0 } ( x ) )$ with $\phi _ { 0 }$ being the maximizer of equation (A.1) and the optimal plan is concentrated where $\phi _ { 0 } ( x ) + \phi _ { 0 } ^ { c } ( y ) = { \textstyle \frac { 1 } { 2 } } d ^ { 2 } ( x , y )$

The relaxation to the Kantorovich problem defines the 2-Wasserstein distance:

$$
{ \frac { 1 } { 2 } } W _ { 2 } ^ { 2 } ( \mu , \nu ) : = \operatorname* { m i n } _ { \pi \in \Pi ( \mu , \nu ) } \int _ { { \mathcal { M } } \times { \mathcal { M } } } { \frac { 1 } { 2 } } d ^ { 2 } ( x , y ) d \pi ( x , y ) ,
$$

where $\Pi ( \mu , \nu )$ is the set of couplings with marginals $\mu$ and $\nu .$ For a regularization parameter $\varepsilon > 0 .$ , the Entropic Optimal Transport objective is:

$$
S _ { \varepsilon } ( \mu , \nu ) : = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } \int _ { \mathcal { M } \times \mathcal { M } } \frac { 1 } { 2 } d ^ { 2 } ( x , y ) d \pi ( x , y ) + \varepsilon D _ { K L } ( \pi \| \mu \otimes \nu ) .\tag{A.2}
$$

The unique solution $\pi _ { \varepsilon }$ has the form $d \pi _ { \varepsilon } ( x , y ) = e ^ { ( f ( x ) + g ( y ) - \frac { 1 } { 2 } d ^ { 2 } ( x , y ) ) / \varepsilon } d \mu ( x ) d \nu ( y )$ where the potentials $f , g \in C ( { \mathcal { M } } )$ solve the dual problem:

$$
\operatorname* { s u p } _ { f , g } \int f d \mu + \int g d \nu - \varepsilon \int _ { \mathcal { M } \times \mathcal { M } } e ^ { \frac { f ( x ) + g ( y ) - \frac { 1 } { 2 } d ^ { 2 } ( x , y ) } { \varepsilon } } d \mu ( x ) d \nu ( y ) + \varepsilon .
$$

Schrödinger bridge. The Schrödinger bridge problem is closely linked to the Entropic Optimal Transport problem. Let $( X _ { t } ) _ { t \in [ 0 , 1 ] }$ denote the Brownian motion on

$( \mathcal { M } , g )$ with generator ${ \frac { \varepsilon } { 2 } } \Delta _ { g } ,$ and let $R _ { \varepsilon }$ be its law on path space $\Omega : = C ( [ 0 , 1 ] , \mathcal { M } )$ Denote by $R _ { \varepsilon } ^ { \bar { 0 } 1 }$ the joint law of the endpoints $( X _ { 0 } , X _ { 1 } )$ under $R _ { \varepsilon } ;$ it admits a density with respect to the product of volume measures given by the heat kernel:

$$
d R _ { \varepsilon } ^ { 0 1 } ( x , y ) = p _ { \varepsilon } ( 1 , x , y ) d \mathrm { v o l } ( x ) d \mathrm { v o l } ( y ) ,
$$

where $p _ { \varepsilon } ( t , x , y )$ solves $\begin{array} { r } { \partial _ { t } p _ { \varepsilon } = \frac { \varepsilon } { 2 } \Delta _ { g } p _ { \varepsilon } } \end{array}$

The Schrödinger bridge problem seeks a coupling $\pi$ that minimizes

$$
C _ { \varepsilon } ( \mu , \nu ) = \operatorname* { i n f } _ { \pi \in \Pi ( \mu , \nu ) } D _ { \mathrm { K L } } ( \pi \| R _ { \varepsilon } ^ { 0 1 } ) .
$$

A.2. The Riemannian Entropic Map. In the Euclidean setting, the entropic map is defined as the barycentric projection $T _ { \varepsilon } ( x ) = \mathbb { E } _ { \pi _ { \varepsilon } } [ Y | X = x ]$ ]. On a manifold, the direct expectation is not well-defined. Instead, we define the map via the Riemannian exponential map and the conditional expectation in the tangent space.

Definition A.1 (Riemannian Entropic Map). Let $( f _ { \varepsilon } , g _ { \varepsilon } )$ be the optimal entropic potentials. We define the Riemannian entropic map $T _ { \varepsilon } : \mathcal { M }  \mathcal { M }$ as:

$$
T _ { \varepsilon } ( x ) : = \exp _ { x } \left( \int _ { \mathcal { M } } \log _ { x } ( y ) d \pi _ { \varepsilon } ^ { x } ( y ) \right) ,\tag{A.3}
$$

where $\log _ { x } = \exp _ { x } ^ { - 1 }$ is the Riemannian logarithm and $\pi _ { \varepsilon } ^ { x }$ is the conditional distribution of Y given $X = x$ under the optimal plan. With some abuse of notation, $\log _ { x }$ and $\exp _ { x }$ are the (Riemannian) logarithm and exponential maps at point x. When no subscript is given, log and exp denote the natural logarithm and exponential functions.

This definition implies a straightforward computational procedure. To evaluate the map at a source point $x ,$ we rely on the optimal entropic coupling $\pi _ { \varepsilon }$ . The procedure involves three steps:

1. Lift to Tangent Space: For the fixed source point x, we compute the Riemannian logarithm $\log _ { x } ( y )$ for every target point $y$ in the support of $\nu .$ This maps the geometry from the manifold M onto the vector space $T _ { x } { \mathcal { M } } .$

2. Euclidean Averaging: We compute the weighted average of these tangent vectors, where the weights correspond to the conditional probability mass $\pi _ { \varepsilon } ( y | x )$ . Because $T _ { x } { \mathcal { M } }$ is a linear space, this is a simple Euclidean average.

3. Retract to Manifold: We apply the exponential map $\mathrm { e x p } _ { x }$ to this average vector to project the result back onto the manifold, yielding the estimated transport location.

We now establish the relationship between this map and the dual potential gradient.

Theorem A.2. Let $( f _ { \varepsilon } , g _ { \varepsilon } )$ be optimal entropic potentials. The primal feasibility $o f \pi _ { \varepsilon }$ requires that its first marginal is $\mu$ . In terms of the potentials, this constraint implies:

$$
\int _ { \mathcal { M } } e ^ { \frac { f _ { \varepsilon } ( x ) + g _ { \varepsilon } ( y ) - \frac { 1 } { 2 } d ^ { 2 } ( x , y ) } { \varepsilon } } d \nu ( y ) = 1 , \quad \forall x \in s u p p ( \mu ) .\tag{A.4}
$$

Under this condition, the Riemannian entropic map satisfies:

$$
T _ { \varepsilon } ( x ) = \exp _ { x } \left( - \nabla f _ { \varepsilon } ( x ) \right) .
$$

Proof. We assume the potentials satisfy (A.4). Taking the logarithm, we isolate $f _ { \varepsilon } ( x )$

$$
f _ { \varepsilon } ( x ) = - \varepsilon \log \int _ { \mathcal { M } } \exp \left( \frac { g _ { \varepsilon } ( y ) - \frac { 1 } { 2 } d ^ { 2 } ( x , y ) } { \varepsilon } \right) d \nu ( y ) .
$$

Let $\begin{array} { r } { h ( x , y ) = \exp \left( \frac { g _ { \varepsilon } ( y ) - \frac { 1 } { 2 } d ^ { 2 } ( x , y ) } { \varepsilon } \right) } \end{array}$ . We compute the Riemannian gradient $\nabla f _ { \varepsilon } ( x )$

$$
\nabla f _ { \varepsilon } ( x ) = - \varepsilon \frac { \nabla _ { x } \left( \int _ { M } h ( x , y ) d \nu ( y ) \right) } { \int _ { M } h ( x , y ) d \nu ( y ) } .
$$

Using the identity $\begin{array} { r } { \nabla _ { x } ( \frac { 1 } { 2 } d ^ { 2 } ( x , y ) ) = - \log _ { x } ( y ) } \end{array}$ , the gradient of the integrand is:

$$
\nabla _ { x } h ( x , y ) = \frac { 1 } { \varepsilon } h ( x , y ) \log _ { x } ( y ) .
$$

Substituting this back, and identifying the conditional density $\begin{array} { r } { d \pi _ { \varepsilon } ^ { x } ( y ) = \frac { h ( x , y ) d \nu ( y ) } { \int h ( x , z ) d \nu ( z ) } , } \end{array}$ we obtain:

$$
\nabla f _ { \varepsilon } ( x ) = - \int _ { \mathcal M } \log _ { x } ( y ) d \pi _ { \varepsilon } ^ { x } ( y ) .
$$

The integral term is exactly the argument of the exponential map in our definition of $T _ { \varepsilon }$ . Thus, $T _ { \varepsilon } ( x ) = \exp _ { x } ( - \nabla f _ { \varepsilon } ( x ) )$ ). □

## A.3. Properties of the Estimator.

Connection to the Fréchet Mean.. The notion of a “center of mass” on a Riemannian manifold is formalized by the Fréchet mean. Given the conditional distribution $\pi _ { \varepsilon } ^ { x }$ of the target Y given $X = x ,$ the true barycentric projection is the point $m ^ { * }$ that minimizes the expected squared distance:

$$
m ^ { * } = \arg \operatorname* { m i n } _ { z \in \mathcal { M } } F ( z ) , \quad \mathrm { w h e r e } \ F ( z ) : = \frac { 1 } { 2 } \int _ { \mathcal { M } } d ^ { 2 } ( z , y ) d \pi _ { \varepsilon } ^ { x } ( y ) .
$$

Finding $m ^ { * }$ generally requires an iterative optimization procedure, as the gradient of this objective is $\begin{array} { r } { \nabla F ( z ) = - \int _ { \mathcal { M } } \log _ { z } ( y ) d \pi _ { \varepsilon } ^ { x } ( y ) } \end{array}$ , leading to the implicit condition $\begin{array} { r } { \int \log _ { m ^ { * } } ( y ) d \pi _ { \varepsilon } ^ { x } ( y ) = 0 } \end{array}$

Our estimator $T _ { \varepsilon } ( x )$ is computed directly as the exponential of the weighted average of tangent vectors, as defined in (A.3). This definition possesses a geometric connection to the true center of mass as the vector we compute, $\begin{array} { r } { v = \int _ { \mathcal { M } } \log _ { x } ( y ) d \pi _ { \varepsilon } ^ { x } ( y ) } \end{array}$ is exactly the negative gradient of the Fréchet objective evaluated at the source point x:

$$
\begin{array} { r } { \boldsymbol { v } = - \nabla F ( \boldsymbol { x } ) . } \end{array}
$$

Consequently, the operation $T _ { \varepsilon } ( x ) = \exp _ { x } ( v )$ geometrically corresponds to taking a single gradient descent step on the objective $F ( z )$ , initialized at x with a step size of 1. We stress that our estimator is the correct generalization of the entropic map to the Riemannian setting, and we found the connection to the Fréchet mean to be a useful geometric intuition.

Recovery of the Euclidean Case.. If we reduce M to the Euclidean space $\mathbb { R } ^ { d }$ equipped with the standard metric, the geometry simplifies: $d ( x , y ) = \| x - y \|$ , the exponential map becomes translation $\exp _ { x } ( v ) = x + v$ , and the logarithm becomes subtraction $\log _ { x } ( y ) = y - x$ . Substituting these into our definition:

$$
T _ { \varepsilon } ( x ) = \exp _ { x } \left( \int ( y - x ) d \pi _ { \varepsilon } ^ { x } ( y ) \right) = x + \left( \int y d \pi _ { \varepsilon } ^ { x } ( y ) - x \int d \pi _ { \varepsilon } ^ { x } ( y ) \right) .
$$

Since $\pi _ { \varepsilon } ^ { x }$ is a probability distribution, it integrates to 1. Thus:

$$
T _ { \varepsilon } ( x ) = \int y d \pi _ { \varepsilon } ^ { x } ( y ) = \mathbb { E } _ { \pi _ { \varepsilon } } [ Y | X = x ] .
$$

This recovers the standard barycentric projection estimator from Pooladian and Niles-Weed [2021].

A.4. McCann Interpolation via the Entropic Map. A fundamental concept in optimal transport geometry is the displacement interpolation, or McCann interpolation, which describes the geodesic path between probability measures in Wasserstein space. Specifically, if $T _ { 0 }$ is the optimal transport map pushing $\mu \ { \mathrm { t o } } \ \nu ,$ the interpolation at time $t \in [ 0 , 1 ]$ is the distribution $\mu _ { t } = ( T _ { 0 , t } ) _ { \# } \mu _ { \# }$ , where $T _ { 0 , t } ( x )$ moves the mass at x a fraction t of the way along the geodesic toward $T _ { 0 } ( x )$

Our Riemannian entropic map ofers a constructive way to approximate this interpolation. Because the map $T _ { \varepsilon } ( x )$ is constructed via the exponential map of a tangent vector $\begin{array} { r } { v _ { x } = \int \log _ { x } ( y ) d \pi _ { \varepsilon } ^ { x } ( y ) } \end{array}$ , the geodesic connecting x to $T _ { \varepsilon } ( x )$ is simply the curve $\gamma ( t ) = \exp _ { x } ( t \cdot v _ { x } )$ .

Definition A.3 (Entropic Interpolant). For any time $t \in [ 0 , 1 ]$ , we define the time-t entropic map $T _ { \varepsilon , t } : \mathcal { M } \to \mathcal { M }$ as:

$$
T _ { \varepsilon , t } ( x ) : = \exp _ { x } \left( t \cdot \int _ { \mathcal { M } } \log _ { x } ( y ) d \pi _ { \varepsilon } ^ { x } ( y ) \right) .
$$

The estimated McCann interpolation at time t is the pushforward measure $\hat { \mu } _ { t } : =$ $( T _ { \varepsilon , t } ) _ { \# } \mu$

This formulation is geometrically intuitive: we compute the aggregate direction of transport $v _ { x }$ in the tangent space and simply scale it by t before projecting back to the manifold. $\mathrm { A t } ~ t = 0$ , the argument to the exponential map is the zero vector, recovering the identity map $( T _ { \varepsilon , 0 } ( x ) = x )$ $\mathrm { A t } ~ t = 1$ , we recover the full Riemannian entropic map $( T _ { \varepsilon , 1 } ( x ) = T _ { \varepsilon } ( x ) )$ . For intermediate t, this generates a distribution supported on the geodesic flow between the source and the estimated target.

A.5. Out-of-Sample Estimation. A critical feature of the proposed estimator is its ability to generalize to points outside the initial training set. Let $\begin{array} { r } { \hat { \nu } = \sum _ { j = 1 } ^ { n } \mathbf { b } _ { j } \delta _ { Y _ { j } } } \end{array}$ be the discrete empirical measure of the target distribution, and let $\mathbf { g } \in \mathbb { R } ^ { n }$ be the optimal dual potential vector obtained from Sinkhorn’s algorithm on training samples.

We seek to evaluate the map $T _ { \varepsilon } ( x ^ { \prime } )$ for a new source point $x ^ { \prime } \in \mathcal { M }$ that was not present during training. This evaluation is not an ad-hoc interpolation but a direct consequence of the extension principle in semi-discrete optimal transport.

The $( c , \varepsilon ) \ – \ T r a n s f o r m$ and Dual Extension.. In classical Optimal Transport $( \varepsilon = 0 )$ )， the relationship between optimal potentials is governed by the c-transform $( \mathrm { o r } ~ c \mathrm { - }$ conjugate), which generalizes the Legendre-Fenchel transform. The source potential $f$ is obtained from the target potential $g$ via a “hard” infimum:

$$
f ( x ) = g ^ { c } ( x ) : = \operatorname* { i n f } _ { y \in \mathcal { M } } \left( \frac { 1 } { 2 } d ^ { 2 } ( x , y ) - g ( y ) \right) .
$$

In Entropic Optimal Transport, this non-smooth operator is replaced by $\mathrm { a \ ^ { 6 6 } s o f t ^ { 9 } }$ minimum known as the $( c , \varepsilon ) \cdot$ -transform:

$$
f _ { \varepsilon } ( x ) = g ^ { ( c , \varepsilon ) } ( x ) : = - \varepsilon \log \left( \int _ { \mathcal { M } } \exp \left( \frac { g ( y ) - \frac { 1 } { 2 } d ^ { 2 } ( x , y ) } { \varepsilon } \right) d \nu ( y ) \right) .
$$

In our estimation setting, we solve the dual problem using the discrete empirical measure $\hat { \nu } .$ While the resulting optimal potential g is a vector in $\mathbb { R } ^ { n }$ , the $( c , \varepsilon ) \cdot$ transform interprets these values as coeficients of a Riemannian kernel expansion. Substituting the discrete measure $\hat { Q }$ into the integral definition yields a globally defined, smooth potential $f _ { \varepsilon } : \mathcal { M } \to \mathbb { R }$

$$
f _ { \varepsilon } ( x ^ { \prime } ) = - \varepsilon \log \left( \sum _ { j = 1 } ^ { n } \mathbf { b } _ { j } \exp \left( \frac { \mathbf { g } _ { j } - \frac { 1 } { 2 } d ^ { 2 } ( x ^ { \prime } , Y _ { j } ) } { \varepsilon } \right) \right) ,\tag{A.5}
$$

where $\mathbf { b } _ { j }$ are the target weights (typically $1 / n )$

Derivation of the Out-of-Sample Map.. We apply our main theorem, $T _ { \varepsilon } ( x ^ { \prime } ) =$ $\exp _ { x ^ { \prime } } ( - \nabla f _ { \varepsilon } ( x ^ { \prime } ) )$ , to this extended potential. Diferentiating (A.5) with respect to $x ^ { \prime }$ involves the gradient of the log-sum-exp function. By the chain rule:

$$
\begin{array} { r l } & { \nabla f _ { \varepsilon } ( x ^ { \prime } ) = - \varepsilon \frac { \sum _ { j = 1 } ^ { n } \nabla _ { x ^ { \prime } } \left[ { \mathbf b } _ { j } \exp \left( \frac { { \mathbf g } _ { j } - \frac { 1 } { 2 } d ^ { 2 } ( x ^ { \prime } , Y _ { j } ) } { \varepsilon } \right) \right] } { \sum _ { k = 1 } ^ { n } { \mathbf b } _ { k } \exp \left( \frac { { \mathbf g } _ { k } - \frac { 1 } { 2 } d ^ { 2 } ( x ^ { \prime } , Y _ { k } ) } { \varepsilon } \right) } } \\ & { \qquad = - \varepsilon \sum _ { j = 1 } ^ { n } \frac { { \mathbf b } _ { j } \exp \left( \frac { { \mathbf g } _ { j } - \frac { 1 } { 2 } d ^ { 2 } ( x ^ { \prime } , Y _ { j } ) } { \varepsilon } \right) } { \sum { \mathbf b } _ { k } \exp \left( \frac { { \mathbf g } _ { k } - \frac { 1 } { 2 } d ^ { 2 } ( x ^ { \prime } , Y _ { k } ) } { \varepsilon } \right) } \cdot \nabla _ { x ^ { \prime } } \left( \frac { { \mathbf g } _ { j } - \frac { 1 } { 2 } d ^ { 2 } ( x ^ { \prime } , Y _ { j } ) } { \varepsilon } \right) . } \end{array}
$$

Since $\mathbf { g } _ { j }$ is constant with respect to $x ^ { \prime } .$ , the inner gradient is simply $\begin{array} { r l } { - { \frac { 1 } { 2 \varepsilon } } \nabla _ { x ^ { \prime } } d ^ { 2 } ( x ^ { \prime } , Y _ { j } ) = } \end{array}$ $\textstyle { \frac { 1 } { \varepsilon } } \log _ { x ^ { \prime } } ( Y _ { j } )$ . Substituting this back yields:

$$
- \nabla f _ { \varepsilon } ( x ^ { \prime } ) = \sum _ { j = 1 } ^ { n } w _ { j } ( x ^ { \prime } ) \log _ { x ^ { \prime } } ( Y _ { j } ) ,
$$

where the attention weights $w _ { j } ( x ^ { \prime } )$ are given by the softmax function:

$$
w _ { j } ( x ^ { \prime } ) = \frac { { \bf b } _ { j } \exp \left( \frac { { \bf g } _ { j } - \frac { 1 } { 2 } d ^ { 2 } ( x ^ { \prime } , Y _ { j } ) } { \varepsilon } \right) } { \sum _ { k = 1 } ^ { n } { \bf b } _ { k } \exp \left( \frac { { \bf g } _ { k } - \frac { 1 } { 2 } d ^ { 2 } ( x ^ { \prime } , Y _ { k } ) } { \varepsilon } \right) } .
$$

Finally, the transport map for the new point $x ^ { \prime }$ is computed by projecting this weighted average back onto the manifold:

$$
T _ { \varepsilon } ( x ^ { \prime } ) = \exp _ { x ^ { \prime } } \left( \sum _ { j = 1 } ^ { n } w _ { j } ( x ^ { \prime } ) \log _ { x ^ { \prime } } ( Y _ { j } ) \right) .
$$

A.6. Computing the Riemannian entropic map without analytic exp and log maps. When closed-form log and exp maps are not available, the entropic estimator remains computationally viable as long as the geodesic distance function $d _ { g } ( \cdot , \cdot )$ on $\mathcal { M }$ is accessible. Specifically, we assume we have access to: (i) the geodesic distance $d _ { g } ( x , y )$ and its extrinsic gradient $\nabla _ { x } d _ { g } ( x , y ) \in \mathbb { R } ^ { d }$ , (ii) a projection operator $\Pi _ { \mathcal { M } } : \mathbb { R } ^ { d }  \mathcal { M }$ mapping ambient points to the closest point on M, and (iii) the Jacobian of the projection operator $J _ { \Pi } ( x ) \in \mathbb { R } ^ { d \times d }$ , which at points $x \in \mathcal { M }$ projects ambient vectors onto the tangent space $T _ { x } { \mathcal { M } }$

Approximating the logarithmic map.. The $N ^ { 2 }$ logarithmic map evaluations in the lift step can be obtained from the gradient of the squared distance. Define $D ( x , y ) =$ $\textstyle { \frac { 1 } { 2 } } d _ { g } ( x , y ) ^ { 2 }$ . Its extrinsic gradient with respect to $x$ is $\nabla _ { x } D ( x , y ) = d _ { g } ( x , y ) \nabla _ { x } d _ { g } ( x , y )$ Projecting onto the tangent space at x via the Jacobian of the manifold projection gives:

$$
\begin{array} { r } { \log _ { x } ( y ) = - J _ { \Pi } ( x ) \big ( d _ { g } ( x , y ) \nabla _ { x } d _ { g } ( x , y ) \big ) . } \end{array}
$$

Substituting into the Riemannian entropic map, the full lift-average-retract computation becomes:

$$
T _ { \varepsilon } ( x _ { i } ) = \exp _ { x _ { i } } \left( - \sum _ { j = 1 } ^ { N } \pi _ { \varepsilon } ( y _ { j } | x _ { i } ) J _ { \Pi } ( x _ { i } ) \big ( d _ { g } ( x _ { i } , y _ { j } ) \nabla _ { x _ { i } } d _ { g } ( x _ { i } , y _ { j } ) \big ) \right) .
$$

Since the Sinkhorn algorithm already requires the cost matrix $\begin{array} { r } { C _ { i j } = \frac { 1 } { 2 } d _ { g } ^ { 2 } ( x _ { i } , y _ { j } ) } \end{array}$ , the distance gradients can be obtained by a backward pass through the same computation at negligible additional cost and the tangent projection can be performed with a Jacobian-vector product.

Approximating the exponential map via projected Euler integration.. The $N$ exponential map evaluations in the retract step can be approximated by numerically integrating the geodesic using only the projection operator. Given $x \in \mathcal { M }$ and $v \in T _ { x } { \mathcal { M } }$ , we seek $y = \exp _ { x } ( v )$ . For a given number of steps $K _ { i }$ , we can approximate this geodesic by iteratively taking small steps in the ambient space and projecting the position and velocity back onto the manifold.

1. Euler step: $\begin{array} { r } { \tilde { y } _ { k + 1 } = y _ { k } + \frac { 1 } { K } v _ { k } } \end{array}$

2. Position projection: $y _ { k + 1 } = \Pi _ { \mathcal { M } } ( \tilde { y } _ { k + 1 } )$

3. Velocity projection: $\tilde { v } _ { k + 1 } = J _ { \Pi } ( y _ { k + 1 } ) v _ { k }$

4. Speed renormalization: $v _ { k + 1 } = \| v \| \cdot \tilde { v } _ { k + 1 } / \| \tilde { v } _ { k + 1 } \|$

Where we start with $y _ { 0 } = x$ and $v _ { 0 } = v$ . The last step is to ensure that the speed of the geodesic is preserved. This procedure requires only the projection operator and its Jacobian-vector product. This formulation opens the door to applying RWEFM on geometries where analytic exp and log maps are unavailable, such as triangulated meshes, learned Riemannian metrics, or implicit surfaces; we demonstrate a proof of concept on a triangulated mesh (the Stanford bunny) in Section 4.2.3.

A.7. Statistical performance of the estimator. We now establish a finitesample guarantee for the Riemannian entropic map.

Throughout this section, we write

$$
c ( x , y ) : = \frac { 1 } { 2 } d _ { g } ^ { 2 } ( x , y ) ,
$$

and let $( \varphi _ { 0 } , \varphi _ { 0 } ^ { c } )$ be a pair of optimal Kantorovich potentials for the unregularized problem. For $x \in \Omega$ , define

$$
x ^ { \star } : = T _ { 0 } ( x ) , \qquad v _ { 0 } ( x ) : = \log _ { x } T _ { 0 } ( x ) ,
$$

and the Kantorovich duality gap

$$
D _ { x } ( y ) : = c ( x , y ) - \varphi _ { 0 } ( x ) - \varphi _ { 0 } ^ { c } ( y ) .\tag{A.6}
$$

By optimality, $D _ { x } ( y ) \geq 0$ and $D _ { x } ( T _ { 0 } ( x ) ) = 0 .$

## A.7.1. Regularity assumptions on M and $T _ { 0 }$ .

Assumption A.4 (Geometry and convexity). There exists a compact, strongly geodesically convex set $\Omega \subset M$ such that supp(µ), supp(ν) ⊂ Ω. For every $x \in \Omega$ the logarithmic map $\log _ { x } : \Omega \to T _ { x } M$ is single-valued and smooth. Moreover, on the relevant tangent balls the exponential map obeys the uniform Lipschitz bound

$$
d _ { g } ( \exp _ { x } ( u ) , \exp _ { x } ( v ) ) \leq C _ { \exp } \Vert u - v \Vert _ { g } , \qquad x \in \Omega .
$$

Assumption A.5 (Density bounds). The measures $\mu$ and ν admit densities $f _ { \mu } , f _ { \nu }$ with respect to Riemannian volume on $\Omega ,$ and

$$
0 < m _ { \rho } \leq f _ { \mu } ( x ) , f _ { \nu } ( x ) \leq M _ { \rho } < \infty , \qquad x \in \Omega .
$$

Assumption A.6 (Regularity and quadratic detachment of the Monge map). The Monge problem for the cost $c ( x , y ) = \textstyle { \frac { 1 } { 2 } } d _ { q } ^ { 2 } ( x , y )$ admits a unique optimal map $T _ { 0 } : \Omega  \Omega$ , which is a difeomorphism. There exist constants $0 < \lambda \le \Lambda < \infty$ such that, uniformly for $x , y \in \Omega$ ，

$$
\frac { \lambda } { 2 } d _ { g } ^ { 2 } ( y , T _ { 0 } ( x ) ) \leq D _ { x } ( y ) \leq \frac { \Lambda } { 2 } d _ { g } ^ { 2 } ( y , T _ { 0 } ( x ) ) .\tag{A.7}
$$

Assumption A.4 implies that c is smooth on the compact set $\Omega \times \Omega$ , hence all derivatives of c that appear below are uniformly bounded. It also implies the following two basic geometric facts. First, there is a constant $C _ { \mathrm { l o g } }$ such that

$$
\Vert \log _ { x } ( y ) - \log _ { x } ( z ) \Vert _ { g } \leq C _ { \log } d _ { g } ( y , z ) , \qquad x , y , z \in \Omega .\tag{A.8}
$$

Second, since Ω is a compact subset of a d-dimensional smooth manifold, it admits measurable partitions into at most $C \delta ^ { - d }$ sets of geodesic diameter at most δ, for every suficiently small $\delta > 0$

Let

$$
\mu _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { X _ { i } } , \qquad \nu _ { n } = \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \delta _ { Y _ { j } } ,
$$

where $X _ { i } \stackrel { \mathrm { i i d } } { \sim } \mu$ and $Y _ { j } \overset { \mathrm { i i d } } { \sim } \nu ,$ with the two samples independent. We denote by $\pi _ { \varepsilon , n }$ the optimal entropic plan between $\mu$ and $\nu _ { n }$ , and by $T _ { \varepsilon , n }$ its Riemannian barycentric map,

$$
T _ { \varepsilon , n } ( x ) : = \exp _ { x } ( b _ { \varepsilon , n } ( x ) ) , \qquad b _ { \varepsilon , n } ( x ) : = \int _ { \Omega } \log _ { x } ( y ) d \pi _ { \varepsilon , n } ^ { x } ( y ) .
$$

For the two-sample estimator, let $( \widehat f , \widehat g )$ be optimal entropic potentials for $( \mu _ { n } , \nu _ { n } )$ We use the canonical out-of-sample extension already described in Section $\mathrm { { A . 5 } } \mathrm { { : } }$ for every $x \in \Omega$ 2,

$$
\widehat { f } ( x ) : = - \varepsilon \log \left[ \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \exp \left( \frac { \widehat { g } ( Y _ { j } ) - c ( x , Y _ { j } ) } { \varepsilon } \right) \right] ,\tag{A.9}
$$

and

(A.10)

$$
\widehat w _ { j } ( x ) : = \frac { \exp ( ( \widehat g ( Y _ { j } ) - c ( x , Y _ { j } ) ) / \varepsilon ) } { \sum _ { k = 1 } ^ { n } \exp ( ( \widehat g ( Y _ { k } ) - c ( x , Y _ { k } ) ) / \varepsilon ) } ,\tag{A.11}
$$

$$
\widehat { b } _ { \varepsilon , ( n , n ) } ( x ) : = \sum _ { j = 1 } ^ { n } \widehat { w } _ { j } ( x ) \log _ { x } ( Y _ { j } ) , \qquad \widehat { T } _ { \varepsilon , ( n , n ) } ( x ) : = \exp _ { x } \Bigl ( \widehat { b } _ { \varepsilon , ( n , n ) } ( x ) \Bigr ) .
$$

At the sampled source points, this agrees with the barycentric projection of the empirical optimal entropic coupling.

Define

$$
s _ { d } : = \operatorname* { m a x } \left\{ 1 , \frac { d } { 2 } \right\} .\tag{A.12}
$$

The use of $s _ { d }$ only matters in dimension one; for every $d \ge 2 , s _ { d } = d / 2$

Theorem A.7 (Statistical performance of the Riemannian entropic map). Under Assumptions $A . 4 { - } A . 6 ,$ there exist constants $C < \infty$ and $\varepsilon _ { 0 } \in ( 0 , 1 ]$ , depending only on the regularity constants and the geometry of Ω, such that $f o r$ every $0 < \varepsilon \le \varepsilon _ { 0 }$ and $n \geq 2$

$$
\mathbb { E } \int _ { \Omega } d _ { g } ^ { 2 } \Big ( \widehat { T } _ { \varepsilon , ( n , n ) } ( x ) , T _ { 0 } ( x ) \Big ) \ d \mu ( x ) \leq C \left[ \varepsilon \log \Big ( \frac { e } { \varepsilon } \Big ) + \varepsilon ^ { - s _ { d } } \frac { \log ( n + 1 ) } { \sqrt { n } } \right] .\tag{A.13}
$$

In particular, for $d \geq 2$

$$
\begin{array} { r } { \mathbb { E } \int _ { \Omega } d _ { g } ^ { 2 } \Big ( \widehat { T } _ { \varepsilon , ( n , n ) } ( x ) , T _ { 0 } ( x ) \Big ) \ d \mu ( x ) \lesssim \varepsilon \log \Big ( \frac { e } { \varepsilon } \Big ) + \varepsilon ^ { - d / 2 } \frac { \log ( n + 1 ) } { \sqrt { n } } . } \end{array}\tag{A.14}
$$

Consequently, choosing $\varepsilon \asymp n ^ { - 1 / ( d + 2 ) }$ for $d \geq 2$ gives, up to logarithmic factors,

$$
\mathbb { E } \int _ { \Omega } d _ { g } ^ { 2 } \Big ( \widehat { T } _ { \varepsilon , ( n , n ) } ( x ) , T _ { 0 } ( x ) \Big ) \ d \mu ( x ) = \widetilde O \Big ( n ^ { - 1 / ( d + 2 ) } \Big ) .
$$

A.7.2. Deterministic and empirical ingredients. We first record a deterministic observation converting the Kantorovich duality gap into map error.

Lemma A.8 (Duality gap controls barycentric map error). Let $\pi \in \Pi ( \mu , \eta )$ be any coupling whose second marginal η is supported in Ω. Define

$$
b _ { \pi } ( x ) : = \int _ { \Omega } \log _ { x } ( y ) d \pi ^ { x } ( y ) , \qquad T _ { \pi } ( x ) : = \exp _ { x } ( b _ { \pi } ( x ) ) .
$$

Then

$$
\int _ { \Omega } d _ { g } ^ { 2 } ( T _ { \pi } ( x ) , T _ { 0 } ( x ) ) d \mu ( x ) \leq C \int _ { \Omega \times \Omega } D _ { x } ( y ) d \pi ( x , y ) .\tag{A.15}
$$

Proof. By Assumption A.4 and Jensen’s inequality,

$$
\begin{array} { r l r } {  { d _ { g } ^ { 2 } ( T _ { \pi } ( x ) , T _ { 0 } ( x ) ) \le C _ { \exp } ^ { 2 } \| b _ { \pi } ( x ) - v _ { 0 } ( x ) \| _ { g } ^ { 2 } } } \\ & { } & { \le C _ { \exp } ^ { 2 } \displaystyle \int _ { \Omega } \| \log _ { x } ( y ) - \log _ { x } ( T _ { 0 } ( x ) ) \| _ { g } ^ { 2 } d \pi ^ { x } ( y ) . } \end{array}
$$

Using (A.8) and the lower bound in (A.7),

$$
\begin{array} { r } { \Vert \log _ { x } ( y ) - \log _ { x } ( T _ { 0 } ( x ) ) \Vert _ { g } ^ { 2 } \leq C d _ { g } ^ { 2 } ( y , T _ { 0 } ( x ) ) \leq C D _ { x } ( y ) . } \end{array}
$$

Integrating first in $y$ and then in x proves the claim.

The next lemma bounds the regularization bias without any heat-kernel representation.

Lemma A.9 (Entropic cost bias). There exists $C < \infty$ such that, for all suficiently small $\varepsilon \in ( 0 , 1 ]$

$$
0 \leq S _ { \varepsilon } ( \mu , \nu ) - \frac 1 2 W _ { 2 } ^ { 2 } ( \mu , \nu ) \leq C \varepsilon \log \left( \frac { e } { \varepsilon } \right) .\tag{A.16}
$$

Proof. The lower bound is immediate because the transport part of the entropic objective is at least the unregularized optimal cost and the relative entropy is nonnegative.

For the upper bound, fix $0 < \delta < 1$ . By compactness of $\Omega ,$ choose a measurable partition $\{ B _ { k } \} _ { k = 1 } ^ { \bar { N } }$ of Ω such that

$$
\begin{array} { r } { \mathrm { d i a m } _ { g } ( B _ { k } ) \le \delta , \qquad N \le C \delta ^ { - d } . } \end{array}
$$

Set $A _ { k } : = T _ { 0 } ^ { - 1 } ( B _ { k } )$ and $p _ { k } : = \mu ( A _ { k } ) = \nu ( B _ { k } )$ . Ignoring cells with $p _ { k } = 0$ , define

$$
\pi ^ { \delta } : = \sum _ { k = 1 } ^ { N } \frac { 1 } { p _ { k } } \mu | _ { A _ { k } } \otimes \nu | _ { B _ { k } } .\tag{A.17}
$$

Then $\pi ^ { \delta } \in \Pi ( \mu , \nu )$ and

$$
D _ { \mathrm { K L } } ( \pi ^ { \delta } \| \mu \otimes \nu ) = \sum _ { k = 1 } ^ { N } p _ { k } \log \frac { 1 } { p _ { k } } \leq \log N \leq C + d \log \frac { 1 } { \delta } .\tag{A.18}
$$

Moreover, if $( x , y ) \in A _ { k } \times B _ { k }$ , then both $T _ { 0 } ( x )$ and $y$ belong to $B _ { k }$ , so the upper bound in (A.7) gives $D _ { x } ( y ) \leq C \delta ^ { 2 }$ . Since $\pi ^ { \delta }$ and the optimal Monge plan have the same marginals, Kantorovich duality yields

$$
\int c d \pi ^ { \delta } - \frac { 1 } { 2 } W _ { 2 } ^ { 2 } ( \mu , \nu ) = \int D _ { x } ( y ) d \pi ^ { \delta } ( x , y ) \leq C \delta ^ { 2 } .\tag{A.19}
$$

Using $\pi ^ { \delta }$ as a competitor in the entropic problem therefore gives

$$
S _ { \varepsilon } ( \mu , \nu ) - \frac { 1 } { 2 } W _ { 2 } ^ { 2 } ( \mu , \nu ) \leq C \delta ^ { 2 } + C \varepsilon + d \varepsilon \log \frac { 1 } { \delta } .
$$

Taking $\delta = \sqrt { \varepsilon }$ proves (A.16).

We next state the empirical-process estimate used twice below. We include the argument because it is also the point at which the intrinsic dimension d enters the statistical rate.

Lemma A.10 (Empirical process bound for entropic transforms). Let $\rho$ be any probability measure supported on $\Omega ,$ and let $\rho _ { n }$ be its empirical measure based on n iid observations. Consider functions of the form

$$
F ( x ) = - \varepsilon \log \int _ { \Omega } \exp \left( \frac { a ( y ) - c ( x , y ) } { \varepsilon } \right) d \xi ( y ) ,\tag{A.20}
$$

where $\xi$ is an arbitrary probability measure on Ω and $a : \Omega \to \mathbb { R }$ is bounded and measurable. Since adding a constant to F does not change $( \rho _ { n } - \rho ) F$ , normalize all

such functions by fixing $F ( x _ { \circ } ) = 0$ at one reference point $x _ { \circ } \in \Omega$ , and denote the resulting class by $\mathcal { F } _ { \varepsilon }$ . Then

$$
\mathbb { E } \operatorname* { s u p } _ { F \in \mathcal { F } _ { \varepsilon } } \left| \int _ { \Omega } F d ( \rho _ { n } - \rho ) \right| \leq C \left( 1 + \varepsilon ^ { 1 - s _ { d } } \right) \frac { \log ( n + 1 ) } { \sqrt { n } } .\tag{A.21}
$$

The constant is uniform over $\rho , \xi ,$ and $a .$

Proof. Because c is smooth on the compact set $\Omega \times \Omega$ , all of its partial derivatives are uniformly bounded. Diferentiating (A.20) shows first that

$$
\nabla F ( x ) = \int _ { \Omega } \nabla _ { x } c ( x , y ) d \omega _ { x } ( y ) ,
$$

where $\omega _ { x }$ is the Gibbs probability measure proportional to $\exp ( ( a ( y ) - c ( x , y ) ) / \varepsilon ) d \xi ( y )$ Repeated diferentiation gives, for every integer $k \geq 1$

$$
\lVert F \rVert _ { C ^ { k } ( \Omega ) } \leq C _ { k } \left( 1 + \varepsilon ^ { 1 - k } \right) .\tag{A.22}
$$

Indeed, the k-th derivative is a finite sum of products of derivatives of c and centered moments under $\omega _ { x } .$ , with at most $k - 1$ powers of $\varepsilon ^ { - 1 }$ . Interpolation between consecutive integer orders gives the corresponding estimate for noninteger Hölder exponents. Thus, with $s _ { d }$ from (A.12),

$$
\operatorname* { s u p } _ { F \in \mathcal { F } _ { \varepsilon } } \| F \| _ { C ^ { s _ { d } } ( \Omega ) } \leq C \left( 1 + \varepsilon ^ { 1 - s _ { d } } \right) .\tag{A.23}
$$

The normalization $F ( x _ { \circ } ) = 0$ , together with the uniform first-derivative bound, also gives a uniform envelope $\| F \| _ { \infty } \leq C$

A finite smooth atlas reduces the metric entropy calculation to the standard one for Hölder balls on bounded subsets of $\mathbb { R } ^ { d }$ [Huang et al., 2022]. Consequently, with $M _ { \varepsilon } = { \cal C } ( 1 + \varepsilon ^ { 1 - s _ { d } } )$

$$
\log N ( \eta , \mathcal { F } _ { \varepsilon } , \| \cdot \| _ { \infty } ) \leq C \left( \frac { M _ { \varepsilon } } { \eta } \right) ^ { d / s _ { d } } .\tag{A.24}
$$

By definition of $s _ { d } ,$ the exponent $d / s _ { d }$ is at most 2. Standard symmetrization followed by the truncated Dudley entropy integral therefore gives

$$
\mathbb { E } \operatorname* { s u p } _ { F \in \mathcal { F } _ { \varepsilon } } | ( \rho _ { n } - \rho ) F | \leq C M _ { \varepsilon } \frac { \log ( n + 1 ) } { \sqrt { n } } .
$$

This is (A.21).

As a direct consequence, the one-sample entropic cost is stable under empirical replacement of one marginal.

Corollary A.11 (One-sample entropic-cost deviation). For $0 < \varepsilon \le 1$ ，

$$
\mathbb { E } | S _ { \varepsilon } ( \mu , \nu _ { n } ) - S _ { \varepsilon } ( \mu , \nu ) | \leq C \varepsilon ^ { - s _ { d } } \frac { \log ( n + 1 ) } { \sqrt { n } } .\tag{A.25}
$$

Proof. Use the semi-dual representation with the first marginal $\mu$ fixed. For a source-side function $f ,$ optimize the dual objective over the target potential to obtain

$$
f ^ { ( c , \varepsilon ) } ( y ) : = - \varepsilon \log \int _ { \Omega } \exp \left( \frac { f ( x ) - c ( x , y ) } { \varepsilon } \right) d \mu ( x ) .
$$

Then

$$
S _ { \varepsilon } ( \mu , \eta ) = \operatorname* { s u p } _ { f } \left\{ \int f d \mu + \int f ^ { ( c , \varepsilon ) } d \eta \right\} .
$$

Hence

$$
| S _ { \varepsilon } ( \mu , \nu _ { n } ) - S _ { \varepsilon } ( \mu , \nu ) | \leq \operatorname* { s u p } _ { f } \left| \int f ^ { ( c , \varepsilon ) } d ( \nu _ { n } - \nu ) \right| .
$$

The transformed functions belong, up to additive constants, to the class in Lemma A.10. Therefore

$$
\mathbb { E } \left| S _ { \varepsilon } ( \mu , \nu _ { n } ) - S _ { \varepsilon } ( \mu , \nu ) \right| \leq C ( 1 + \varepsilon ^ { 1 - s _ { d } } ) \frac { \log ( n + 1 ) } { \sqrt { n } } .
$$

Since $0 < \varepsilon \le 1$ and $s _ { d } \geq 1$ , the right-hand side is bounded by the one in (A.25).

For completeness, we also record the modified duality inequality needed to compare the one- and two-sample maps. Importantly, it is valid for any cost function and does not use Euclidean linear structure.

Lemma A.12 (Modified entropic duality). Let $P , Q$ be probability measures on $\Omega$ let $\pi _ { \varepsilon }$ be their optimal entropic plan, and let c be any bounded measurable cost. Then

$$
S _ { \varepsilon } ( P , Q ) \geq \int \eta d \pi _ { \varepsilon } - \varepsilon \iint \exp \left( \frac { \eta ( x , y ) - c ( x , y ) } { \varepsilon } \right) d P ( x ) d Q ( y ) + \varepsilon\tag{A.26}
$$

for every $\eta \in L ^ { 1 } ( \pi _ { \varepsilon } )$

Proof. Let $\gamma = d \pi _ { \varepsilon } / d ( P \otimes Q )$ . The elementary inequality

$$
a \log a \geq a b - e ^ { b } + a , \qquad a \geq 0 , \ b \in \mathbb { R } ,
$$

applied with

$$
b = \frac { \eta ( x , y ) - c ( x , y ) } { \varepsilon }
$$

and integrated against $P \otimes Q$ gives the claim after using

$$
S _ { \varepsilon } ( P , Q ) = \int c d \pi _ { \varepsilon } + \varepsilon \int \log \gamma d \pi _ { \varepsilon } .
$$

Proposition A.13 (Stability to empirical sampling of the source). Let $T _ { \varepsilon , n }$ be the entropic map from µ to $\nu _ { n }$ , and let $\widehat { T } _ { \varepsilon , ( n , n ) }$ be the out-of-sample extension in $\mathrm { ( A . 1 1 ) }$ Then, for $0 < \varepsilon \le 1$

$$
\mathbb { E } \int _ { \Omega } d _ { g } ^ { 2 } \Big ( \widehat { T } _ { \varepsilon , ( n , n ) } ( x ) , T _ { \varepsilon , n } ( x ) \Big ) d \mu ( x ) \leq C \varepsilon ^ { - s _ { d } } \frac { \log ( n + 1 ) } { \sqrt { n } } .\tag{A.27}
$$

Proof. Let $( f _ { \varepsilon , n } , g _ { \varepsilon , n } )$ be optimal entropic potentials for $( \mu , \nu _ { n } )$ , normalized so that

$$
\int _ { \Omega } \exp \left( \frac { f _ { \varepsilon , n } ( x ) + g _ { \varepsilon , n } ( y ) - c ( x , y ) } { \varepsilon } \right) d \nu _ { n } ( y ) = 1
$$

for every $x \in \Omega$ . Likewise, by the definition (A.9),

$$
{ \widehat { \gamma } } ( x , y ) : = \exp \left( { \frac { { \widehat { f } } ( x ) + { \widehat { g } } ( y ) - c ( x , y ) } { \varepsilon } } \right)
$$

satisfies

$$
\int _ { \Omega } { \widehat { \gamma } } ( x , y ) d \nu _ { n } ( y ) = 1 \qquad { \mathrm { f o r ~ e v e r y ~ } } x \in \Omega .\tag{A.28}
$$

On the support of $\mu _ { n } \otimes \nu _ { n } , \widehat { \gamma }$ is the density of the empirical optimal entropic plan. In addition,

$$
{ \widehat { b } } _ { \varepsilon , ( n , n ) } ( x ) = \int _ { \Omega } \log _ { x } ( y ) { \widehat { \gamma } } ( x , y ) d \nu _ { n } ( y ) .
$$

Apply Lemma A.12 to the optimal plan $\pi _ { \varepsilon , n }$ between $\mu$ and $\nu _ { n } .$ , with

$$
\eta ( x , y ) = \varepsilon \chi ( x , y ) + \widehat { f } ( x ) + \widehat { g } ( y ) .
$$

Using (A.28) gives, for every integrable $\chi ,$

$$
\begin{array} { r l r } {  { \int \chi d \pi _ { \varepsilon , n } - \iint ( e ^ { \chi ( x , y ) } - 1 ) \widehat { \gamma } ( x , y ) d \mu ( x ) d \nu _ { n } ( y ) } } \\ & { } & { \leq \varepsilon ^ { - 1 } [ S _ { \varepsilon } ( \mu , \nu _ { n } ) - \int \widehat { f } d \mu - \int \widehat { g } d \nu _ { n } ] . } \end{array}\tag{A.29}
$$

We now control the right-hand side. Because $( \widehat f , \widehat g )$ is optimal for $( \mu _ { n } , \nu _ { n } )$ , while $( f _ { \varepsilon , n } , g _ { \varepsilon , n } )$ is an admissible dual pair for that same empirical problem,

$$
\int \widehat { f } d \mu _ { n } + \int \widehat { g } d \nu _ { n } \geq \int f _ { \varepsilon , n } d \mu _ { n } + \int g _ { \varepsilon , n } d \nu _ { n } .
$$

Here no exponential correction remains because both source potentials are chosen as the entropic $( c , \varepsilon )$ -transform of their corresponding target potential, so their exponential term integrates to one pointwise in x. Also,

$$
S _ { \varepsilon } ( \mu , \nu _ { n } ) = \int f _ { \varepsilon , n } d \mu + \int g _ { \varepsilon , n } d \nu _ { n } .
$$

Therefore

$$
\begin{array} { r l } { S _ { \varepsilon } ( \mu , \nu _ { n } ) - \displaystyle \int \widehat { f } d \mu - \int \widehat { g } d \nu _ { n } } & { } \\ { \leq \displaystyle \int ( f _ { \varepsilon , n } - \widehat { f } ) d ( \mu - \mu _ { n } ) . } \end{array}\tag{A.30}
$$

Conditionally on $\nu _ { n }$ , the function $f _ { \varepsilon , n }$ is independent of $\mu _ { n }$ , and hence its contribution has expectation zero. The random function ${ \widehat { f } } ,$ after an irrelevant additive normalization, belongs pathwise to the class $\mathcal { F } _ { \varepsilon }$ from Lemma A.10. Taking expectations in (A.29)– (A.30) therefore gives

$$
\begin{array} { r l } {  { \mathbb { E } \operatorname* { s u p } _ { x } \Bigg \{ \int \chi d \pi _ { \varepsilon , n } - \iint ( e ^ { \chi } - 1 ) \widehat { \gamma } d \mu d \nu _ { n } \Bigg \} } \quad } & { } \\ & { \leq C \varepsilon ^ { - 1 } ( 1 + \varepsilon ^ { 1 - s _ { d } } ) \frac { \log ( n + 1 ) } { \sqrt { n } } } \\ & { \leq C \varepsilon ^ { - s _ { d } } \frac { \log ( n + 1 ) } { \sqrt { n } } . } \end{array}\tag{A.31}
$$

It remains to extract the map error from this functional inequality. For a measurable tangent vector field $h ( x ) \in T _ { x } M$ , set

$$
\chi _ { h } ( x , y ) : = \Big \langle h ( x ) , \log _ { x } ( y ) - \widehat { b } _ { \varepsilon , ( n , n ) } ( x ) \Big \rangle _ { g } - a \| h ( x ) \| _ { g } ^ { 2 } ,\tag{A.32}
$$

where $a > 0$ is a suficiently large geometric constant. Because Ω is compact and $\log _ { x } ( y )$ is uniformly bounded on $\Omega \times \Omega$ , Hoefding’s lemma applied conditionally in x shows that a can be chosen so that

$$
\int _ { \Omega } ( e ^ { \chi _ { h } ( x , y ) } - 1 ) \widehat { \gamma } ( x , y ) d \nu _ { n } ( y ) \leq 0 \qquad \mathrm { f o r ~ e v e r y ~ } x \in \Omega .\tag{A.33}
$$

On the other hand, disintegrating $\pi _ { \varepsilon , n }$ gives

$$
\int \chi _ { h } d \pi _ { \varepsilon , n } = \int _ { \Omega } \left[ \left. h ( x ) , b _ { \varepsilon , n } ( x ) - \widehat { b } _ { \varepsilon , ( n , n ) } ( x ) \right. _ { g } - a \| h ( x ) \| _ { g } ^ { 2 } \right] d \mu ( x ) .
$$

Taking the pointwise supremum over h yields

$$
\operatorname* { s u p } _ { h } \int \chi _ { h } d \pi _ { \varepsilon , n } = \frac { 1 } { 4 a } \int _ { \Omega } \| b _ { \varepsilon , n } ( x ) - \widehat { b } _ { \varepsilon , ( n , n ) } ( x ) \| _ { g } ^ { 2 } d \mu ( x ) .\tag{A.34}
$$

Combining (A.31)–(A.34) and finally applying the Lipschitz bound for the exponential map from Assumption A.4 proves (A.27). □

A.7.3. Proof of Theorem A.7. We first bound the one-sample error. Applying Lemma A.8 to $\pi _ { \varepsilon , n }$ gives

$$
\int d _ { g } ^ { 2 } ( T _ { \varepsilon , n } ( x ) , T _ { 0 } ( x ) ) d \mu ( x ) \leq C \int D _ { x } ( y ) d \pi _ { \varepsilon , n } ( x , y ) .\tag{A.35}
$$

Since $\pi _ { \varepsilon , n }$ is optimal for the entropic problem between $\mu$ and $\nu _ { n }$

$$
\int D _ { x } ( y ) d \pi _ { \varepsilon , n } ( x , y ) \leq S _ { \varepsilon } ( \mu , \nu _ { n } ) - \int \varphi _ { 0 } d \mu - \int \varphi _ { 0 } ^ { c } d \nu _ { n } .\tag{A.36}
$$

Indeed, the diference between the right-hand side and the left-hand side is exactly $\varepsilon D _ { \mathrm { K L } } ( \pi _ { \varepsilon , n } \Vert \mu \otimes \nu _ { n } )$ , which is nonnegative. Taking expectations and using $\mathbb { E } \nu _ { n } = \nu$ together with Kantorovich duality yields

(A.37)

$$
\mathbb { E } \int D _ { x } ( y ) d \pi _ { \varepsilon , n } ( x , y ) \leq \mathbb { E } \left[ S _ { \varepsilon } ( \mu , \nu _ { n } ) - S _ { \varepsilon } ( \mu , \nu ) \right]\tag{A.38}
$$

$$
+ S _ { \varepsilon } ( \mu , \nu ) - \frac { 1 } { 2 } W _ { 2 } ^ { 2 } ( \mu , \nu )\tag{A.39}
$$

$$
\leq C \varepsilon ^ { - s _ { d } } \frac { \log ( n + 1 ) } { \sqrt { n } } + C \varepsilon \log \left( \frac { e } { \varepsilon } \right) ,
$$

where the last line follows from Corollary A.11 and Lemma A.9. Combining (A.35) and (A.39),

$$
\mathbb { E } \int d _ { g } ^ { 2 } ( T _ { \varepsilon , n } ( x ) , T _ { 0 } ( x ) ) d \mu ( x ) \leq C \left[ \varepsilon \log \left( \frac { e } { \varepsilon } \right) + \varepsilon ^ { - s _ { d } } \frac { \log ( n + 1 ) } { \sqrt { n } } \right] .\tag{A.40}
$$

Finally, the squared triangle inequality and Proposition A.13 give

$$
\begin{array} { r l } {  { \mathbb { E } \int d _ { g } ^ { 2 } ( \widehat { T } _ { \varepsilon , ( n , n ) } ( x ) , T _ { 0 } ( x ) ) d \mu ( x ) } } \\ & { \leq 2 \mathbb { E } \int d _ { g } ^ { 2 } ( \widehat { T } _ { \varepsilon , ( n , n ) } ( x ) , T _ { \varepsilon , n } ( x ) ) d \mu ( x ) + 2 \mathbb { E } \int d _ { g } ^ { 2 } ( T _ { \varepsilon , n } ( x ) , T _ { 0 } ( x ) ) d \mu ( x ) } \\ & { \leq C [ \varepsilon \log ( \frac { e } { \varepsilon } ) + \varepsilon ^ { - s _ { d } } \frac { \log ( n + 1 ) } { \sqrt { n } } ] , } \end{array}
$$

which proves Theorem $_ \mathrm { A . 7 } ^ { }$

Appendix B. A short introduction to flow matching. In this section, we provide a very concise introduction to flow matching and highlight the theoretical steps needed to show the validity of the procedure on the space of probability distributions. We refer the reader to Lipman et al. [2024] for a complete introduction to flow matching.

B.0.1. Vector fields generating probability paths. At its core, flow matching aims to find a tractable way to generate samples from a target distribution $\nu$ from a source distribution $\mu .$ One way to do so is by learning a vector field $u _ { t }$ that generates a probability path with the right boundary conditions. By generation, we understand the Lagrangian interpretation. That is, $u _ { t }$ generates a probability path $\mu _ { t } \ \mathrm { i f f } .$ for all $t \in [ 0 , 1 ]$

$$
\begin{array} { r } { \partial _ { t } \phi _ { t } ( x ) = u _ { t } ( \phi _ { t } ( x ) ) \quad } \\ { X _ { t } = \phi _ { t } ( X _ { 0 } ) \sim \mu _ { t } . } \end{array}
$$

With such a $u _ { t }$ , one can then sample from $\mu$ and integrate over time to generate samples from distributions $\mu _ { t }$ . An important result to verify that $u _ { t }$ generates a given probability path is the mass conservation formula:

$$
u _ { t } \ \mathrm { g e n e r a t e s } \ \mu _ { t } \Leftrightarrow \partial _ { t } \mu _ { t } ( x ) + \nabla \cdot ( \mu _ { t } \boldsymbol { u } _ { t } ) ( x ) = 0
$$

That is, $( u _ { t } , \mu _ { t } )$ solving the continuity equation is equivalent to $u _ { t }$ generating $\mu _ { t }$ . However, directly using this relation is impractical as we typically don’t have information about the target distribution $\nu .$

B.0.2. Conditional vector fields and probability paths. A more promising strategy is to construct probability paths using conditional probability paths:

$$
\mu _ { t } ( x ) = \int \mu _ { t | z } ( x | z ) d \pi ( z )
$$

Of course, one needs to design the conditional probability paths and marginal distribution for the conditioning variable such that the boundaries coincide with the constraints $( \mu$ and $\nu )$

If $\mu$ is known (e.g. a standard normal), one can choose

$$
d \pi ( z ) = d \nu ( x _ { 1 } ) , \quad p _ { t | x _ { 1 } } = \mathcal { N } ( t x _ { 1 } , ( 1 - t ) ^ { 2 } )
$$

We verify that in that case:

$$
\begin{array} { l } { \displaystyle { \int \mathcal { N } ( x ; 0 , 1 ) d \nu ( x _ { 1 } ) = \mathcal { N } ( 0 , 1 ) } } \\ { \displaystyle { \int \delta _ { x _ { 1 } } d \nu ( x _ { 1 } ) = \nu } } \end{array}
$$

When $\mu$ is an arbitrary (unknown) distribution, one can instead choose

$$
\begin{array} { l } { { d \pi ( z ) = d \pi ( x _ { 0 } , x _ { 1 } ) , \quad \displaystyle \int d \pi ( x _ { 0 } , x _ { 1 } ) d x _ { 1 } = d \mu ( x _ { 0 } ) , } } \\ { { \displaystyle \int d \pi ( x _ { 0 } , x _ { 1 } ) d x _ { 0 } = d \nu ( x _ { 1 } ) , \quad p _ { t | x _ { 0 } , x _ { 1 } } = \delta _ { ( 1 - t ) x _ { 0 } + t x _ { 1 } } ( x ) } } \end{array}
$$

In which case, we verify again that the boundary conditions are respected

$$
\begin{array} { c } { { \displaystyle \iint \delta _ { x _ { 0 } } d \pi ( x _ { 0 } , x _ { 1 } ) = \int \delta _ { x _ { 0 } } d \mu ( x _ { 0 } ) = \mu ( x ) } } \\ { { \displaystyle \iint \delta _ { x _ { 1 } } d \pi ( x _ { 0 } , x _ { 1 } ) = \int \delta _ { x _ { 1 } } d \nu ( x _ { 1 } ) = \nu ( x ) } } \end{array}
$$

Given the simple form of the conditional probability paths, one can easily derive corresponding vector fields that generate the conditional distributions, $v _ { t } ( x \mid z )$

The first crucial realization of flow matching is that the marginal conditional vector field $v _ { t }$ also generates the marginal probability path $\mu _ { t }$

$$
\partial _ { t } \mu _ { t } ( x ) + \nabla \cdot ( \mu _ { t } v _ { t } ) ( x ) = 0
$$

where

$$
\begin{array} { l l l } & { v _ { t } ( x ) = \displaystyle \int v _ { t } ( x \mid z ) \frac { \mu _ { t \mid z } ( x \mid z ) \pi ( z ) } { \mu _ { t } ( x ) } } \\ & { v _ { t } ( x ) = \mathbb { E } [ v _ { t } ( X _ { t } \mid Z ) \vert X _ { t } = x ] } \end{array}
$$

That is, from simple conditional vector fields that generate simple conditional probability paths, we can construct the marginal vector field that generates the probability path of interest. Now, of course, the equation above is not tractable as $\textstyle P ( Z \mid X _ { t } )$ is generally not available. The second realization of flow matching is that learning $v _ { t } ^ { \theta } ( x )$ by regressing it to conditional vector fields has the same minimizer than regressing $v _ { t } ^ { \theta } ( x )$ against its true value, as shown next.

B.0.3. Flow matching loss. Our intended flow matching loss is to regress a learnable vector field to the marginal vector field:

$$
\mathcal { L } _ { F M } = \mathbb { E } _ { t , X _ { t } } [ \| v _ { t } ( X _ { t } ) - v _ { t } ^ { \theta } ( X _ { t } ) \| ^ { 2 } ]
$$

Instead we can use a loss that involves only tractable distributions:

$$
\mathcal { L } _ { C F M } = \mathbb { E } _ { t , Z \sim \pi ( Z ) , X _ { t } \sim \mu _ { t } ( X _ { t } \mid Z ) } [ \| v _ { t } ( X _ { t } \vert Z ) - v _ { t } ^ { \theta } ( X _ { t } ) \| ^ { 2 } ]
$$

Crucially, one can show that the gradients of both losses are identical.

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { C F M } = \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { F M }
$$

B.0.4. Roadmap to lifting to empirical distributions. In this work, we want to leverage the same strategy for training our generative model but operating on the space of probability distributions defined on a Riemannian manifold M: $\mathcal { P } _ { 2 } ( \mathcal { M } )$ which is an infinite dimensional space. For this to be possible, we need to show the three following statements apply on that space:

• (C1) The existence of an equivalent of the mass conservation formula on $\mathcal { P } _ { 2 } ( \mathcal { P } _ { 2 } ( \mathcal { M } ) )$ (Section C.1)

• (C2) Showing that the marginal vector field $v _ { t }$ generates $\mu _ { t }$ (Section C.2)

• (C3) The gradient of the conditional flow matching loss is identical to the gradient of the flow matching loss (Section C.3)

Appendix C. Lifting Flow Matching to $\mathcal { P } _ { 2 } ( \mathcal { M } )$

Setup and Definitions. We work on the Wasserstein space $\mathcal { P } _ { 2 } ( \mathcal { M } )$ , the space of probability measures on a manifold M with finite second moments. We endow $\mathcal { P } _ { 2 } ( \mathcal { M } )$ with the Borel σ−algebra generated by the $W _ { 2 }$ metric (with the underlying geodesic distance).

## C.1. Continuity equation and Lagrangian flows on $\mathcal { P } _ { 2 } ( \mathcal { M } )$

Objectives of this section. In the following, we show that if $( \mathbb { P } _ { t } , V _ { t } )$ solve the continuity equation on $\mathcal { P } _ { 2 } ( \mathcal { M } ) ( i . e .$ with test functionals on $\mathcal { P } _ { 2 } ( \mathcal { M } ) )$ , then $V _ { t }$ generates $\mathbb { P } _ { t }$ in the Lagrangian sense $\mathbb { P } _ { t } = ( \Phi _ { t } ) _ { \# } \mathbb { P } _ { 0 }$ . That is, conceptually, one can generate the probability path $\mathbb { P } _ { t }$ by flowing individual elements from 0 to t according to the vector field $V _ { t }$ , which underlies the generation procedure used in flow matching.

## C.1.1. Definitions.

Tangent space and path space of continuous curves. For each $\mu \in \mathscr { P } _ { 2 } ( \mathcal { M } )$ , we identify the tangent space $T _ { \mu } \mathcal { P } _ { 2 } ( \mathcal { M } )$ with the closure of the gradient vector fields in $L ^ { 2 } ( \mu ; T \mathcal { M } )$ and endow it with the norm $\begin{array} { r } { \| v \| _ { L ^ { 2 } ( \mu ) } ^ { 2 } = \int _ { \mathcal { M } } \| v ( x ) \| _ { g } ^ { 2 } d \mu ( x ) } \end{array}$ . We further write $L ^ { 2 } ( \mathbb { P } _ { t } , T \mathcal { P } _ { 2 } ( \mathcal { M } ) )$ the space of functions $V _ { t } : \mathcal { P } _ { 2 } ( \mathcal { M } ) \to T \mathcal { P } _ { 2 } ( \mathcal { M } )$ such that $\begin{array} { r } { \int _ { \mathcal { P } _ { 2 } ( \mathcal { M } ) } \| V _ { t } ( \mu ) \| _ { L ^ { 2 } ( \mu ) } ^ { 2 } ( \mu ) d \mathbb { P } _ { t } < \infty } \end{array}$ . We write $\Gamma _ { T } : = C ( [ 0 , T ] ; \mathcal { P } _ { 2 } ( \mathcal { M } ) )$ the path space of continuous curves γ from [0, T] to $\mathcal { P } _ { 2 } ( \mathcal { M } )$

We define the elevation maps $e _ { t }$ as

$$
e _ { t } : ( \gamma ) \in \Gamma _ { T } \to \gamma ( t ) \in \mathcal { P } _ { 2 } ( \mathcal { M } ) \quad \mathrm { f o r } ~ t \in [ 0 , T ]
$$

Cylinder Functions.

Definition C.1. A functional $\mathscr { F } : \mathscr { P } _ { 2 } ( \mathcal { M } )  \mathbb { R }$ is called a cylinder function if there exists an integer $k \geq 1$ , a smooth function with compact support $F \in C _ { c } ^ { \infty } ( \mathbb { R } ^ { k } )$ and a set of smooth functions with compact support $\{ V _ { 1 } , \dots , V _ { k } \} \subset C _ { c } ^ { \infty } ( { \mathcal { M } } )$ , such that for any measure $\mu \in \mathcal P _ { 2 } ( \mathcal M )$

$$
\mathcal { F } ( \mu ) = F \left( \int _ { \mathcal { M } } V _ { 1 } d \mu , \dots , \int _ { \mathcal { M } } V _ { k } d \mu \right)
$$

This definition can be extended to time-dependent functionals $\varphi _ { t } ( \mu ) = \varphi ( t , \mu )$ where $\begin{array} { r } { \varphi ( t , \mu ) = F ( t , \int V _ { 1 } d \mu , \dots , \int V _ { k } d \mu ) } \end{array}$ for some $F \in C _ { c } ^ { \infty } ( I \times \mathbb { R } ^ { k } )$

In this text, we will assume that time-dependent cylinder functions have a compact support in time (0, T), such that $\varphi _ { 0 } ( x ) = \varphi _ { T } ( x ) = 0$ for all $x .$

Using the chain rule, the gradient of a cylinder functional ${ \mathcal F } .$ , the Wasserstein gradient, at a measure $\mu$ can be computed.

Definition C.2. The Wasserstein gradient of a cylinder function ${ \mathcal { F } } _ { z }$ , denoted $\nabla _ { \boldsymbol { \mathcal { W } } } \mathcal { F } ( \boldsymbol { \mu } )$ , is the vector field on $\mathcal { M }$

$$
\nabla _ { \mathcal W } \mathcal F ( \mu ) = \sum _ { i = 1 } ^ { k } \frac { \partial F } { \partial x _ { i } } \left( \int V _ { 1 } d \mu , \dots , \int V _ { k } d \mu \right) \nabla V _ { i }
$$

Here, $\frac { \partial F } { \partial x _ { i } }$ is the partial derivative of $F$ with respect to its i-th argument, and $\nabla V _ { i }$ is the gradient of the function $V _ { i }$ on the manifold $\mathcal { M }$

C.1.2. Continuity equation and superposition principle on $\mathcal { P } _ { 2 } ( \mathcal { M } )$ . We give precise suficient hypotheses for the Lagrangian interpretation used in the main text. The argument adapts the superposition principle for random measures of Pinzi and Savaré [2025, Theorem 1.2] through a smooth embedding.

Recall that $( \mathbb { P } _ { t } , V _ { t } )$ satisfies the weak continuity equation on $\mathcal { P } _ { 2 } ( \mathcal { M } )$ if

(Weak CE)

$$
\begin{array} { r l } {  { \int _ { 0 } ^ { T } \int _ { \mathcal P _ { 2 } ( \mathcal M ) } \Bigl [ \partial _ { t } \varphi _ { t } ( \mu ) } } \\ & { +  \nabla _ { \mathcal W } \varphi _ { t } ( \mu ) , V _ { t } ( \mu )  _ { L ^ { 2 } ( \mu ) } \Bigr ] d \mathbb P _ { t } ( \mu ) d t = 0 , } \end{array}
$$

for every smooth cylinder functional $\varphi _ { t }$ with compact support in time in $( 0 , T )$

Theorem C.3 (Weak continuity equation and superposition in $\mathcal { P } _ { 2 } ( \mathcal { M } ) )$ . Let $( \mathcal { M } , g )$ be a connected, complete, smooth Riemannian manifold without boundary, and let $\left( \mathbb { P } _ { t } \right) _ { t \in [ 0 , T ] }$ be an absolutely continuous curve in $\mathcal { P } _ { 2 } ( \mathcal { P } _ { 2 } ( \mathcal { M } ) )$ . Assume there is a compact set $K \subset { \mathcal { M } }$ such that $\mathbb { P } _ { t } ( \mathcal { P } ( K ) ) = 1$ for every t, where ${ \mathcal { P } } ( K )$ denotes the probability measures supported in $K$ . Let $V _ { t } ( \mu ) \in T _ { \mu } \mathcal { P } _ { 2 } ( \mathcal { M } )$ admit a jointly Borel representative $b ( t , x , \mu ) = V _ { t } ( \mu ) ( x ) \in T _ { x } \mathcal { M }$ , and assume

$$
\int _ { 0 } ^ { T } \int _ { { \mathcal { P } } _ { 2 } ( \mathcal { M } ) } \| V _ { t } ( \mu ) \| _ { L ^ { 2 } ( \mu ) } ^ { 2 } d \mathbb { P } _ { t } ( \mu ) d t < \infty .
$$

Suppose $( \mathbb { P } _ { t } , V _ { t } )$ satisfies the weak continuity equation (Weak CE). Then:

1. There exists a probability measure Γ on $\Gamma _ { T } = C ( [ 0 , T ] ; \mathcal { P } _ { 2 } ( \mathcal { M } ) )$ , concentrated on $W _ { 2 }$ -absolutely continuous curves taking values in ${ \mathcal { P } } ( K )$ , such that

$$
( e _ { t } ) _ { \# } \Gamma = \mathbb { P } _ { t } , \qquad t \in [ 0 , T ] .
$$

2. For Γ-almost every trajectory $\gamma ,$ the measure-valued continuity equation

(Measure CE)

$$
\begin{array} { r } { \partial _ { t } \gamma ( t ) + \mathrm { d i v } _ { \mathcal { M } } \big ( \gamma ( t ) V _ { t } ( \gamma ( t ) ) \big ) = 0 } \end{array}
$$

holds weakly on M. Consequently, for each smooth time-dependent cylinder functional $\varphi _ { t } ,$ along Γ-almost every trajectory and for almost every $t ,$

$$
\frac { d } { d t } \varphi _ { t } ( \gamma ( t ) ) = \partial _ { t } \varphi _ { t } ( \gamma ( t ) ) + \big \langle \nabla _ { \mathcal { W } } \varphi _ { t } ( \gamma ( t ) ) , V _ { t } ( \gamma ( t ) ) \big \rangle _ { L ^ { 2 } ( \gamma ( t ) ) } .
$$

This is the weak interpretation of ${ \dot { \gamma } } ( t ) = V _ { t } ( \gamma ( t ) )$ used here.

3. Assume additionally that, for P<sub>0</sub>-almost every $\mu ,$ , equation (Measure CE) has at most one $W _ { 2 } – a b s o l u t e l y$ continuous solution in ${ \mathcal { P } } ( K )$ with $\gamma ( 0 ) = \mu$ . Then, for P<sub>0</sub>-almost every $\mu ,$ there exists a unique W<sub>2</sub>-absolutely continuous curve $\gamma _ { \mu } : [ 0 , T ] \to { \mathcal { P } } ( K ) \subset { \mathcal { P } } _ { 2 } ( { \mathcal { M } } )$ solving

$$
\begin{array} { r l } & { \dot { \gamma } _ { \mu } ( t ) = V _ { t } ( \gamma _ { \mu } ( t ) ) , \qquad f o r \ a . e . \ t \in [ 0 , T ] , } \\ & { \gamma _ { \mu } ( 0 ) = \mu , } \end{array}
$$

where the evolution equation is understood in the weak sense of (Measure CE), equivalently through the cylinder chain rule above. The map $\Phi _ { t } ( \mu ) : = \gamma _ { \mu } ( t )$ defines a measurable flow, up to $\mathbb { P } _ { 0 }$ -null sets, such that

$$
\mathbb { P } _ { t } = ( \Phi _ { t } ) _ { \# } \mathbb { P } _ { 0 } , \qquad t \in [ 0 , T ] .
$$

Thus $V _ { t }$ generates the probability path $\mathbb { P } _ { t }$ by evolving each initial measure along its trajectory $\gamma _ { \mu }$

Proof. Choose a smooth embedding $\ j : \mathcal { M } \to \mathbb { R } ^ { D }$ and write $J ( \mu ) = \ j _ { \# } \mu$ for $\mu \in { \mathcal { P } } ( K )$ . Set $\widetilde { \mathbb { P } } _ { t } = J _ { \# } \mathbb { P } _ { t }$ and define

$$
\widetilde { b } ( t , \jmath ( x ) , J ( \mu ) ) = d \jmath _ { x } b ( t , x , \mu ) , \qquad x \in { \cal K } , \quad \mu \in \mathcal { P } ( { \cal K } ) ,
$$

extending $\widetilde { b }$ by zero elsewhere. $\mathrm { O n }$ the compact set $K$ , the embedding has bounded diferential and its inverse has bounded metric distortion. Pulling back cylinder tests, using a smooth cutof equal to one near $K$ , transfers the weak continuity equation to $( \widetilde { \mathbb { P } } _ { t } , \widetilde { \boldsymbol { b } } )$ on ${ \mathcal { P } } ( \mathbb { R } ^ { D } )$ . The integrated squared-velocity bound is preserved up to a constant and implies the integrability required by Pinzi and Savaré [2025, Theorem 1.2].

That theorem gives a measure $\widetilde \Gamma$ on measure-valued trajectories with marginals $\widetilde { \mathbb { P } } _ { t } ,$ concentrated on solutions of the continuity equation driven by ${ \widetilde { b } } .$ These trajectories remain in ${ \mathcal { P } } ( { \boldsymbol { \jmath } } ( K ) )$ : the marginal identities imply this simultaneously at rational times, and continuity and closedness extend it to every time. We can therefore pull them back through $J ^ { - 1 }$ to obtain Γ with the desired marginals. Extension of smooth tests from the embedded manifold and the chain rule give (Measure CE) [Pinzi, 2026].

By Fubini and the marginal identities, the assumed energy is finite along almost every represented trajectory. The velocity-energy estimate for the Euclidean continuity equation gives $W _ { 2 }$ absolute continuity there; metric comparison on $K$ transfers this to the intrinsic $W _ { 2 }$ metric. Applying the cylinder chain rule to (Measure CE) proves the second claim.

Disintegration and the deterministic flow.. For the third claim, consider the joint law of the initial measure and the entire trajectory:

$$
\widehat { \Gamma } : = ( e _ { 0 } , \mathrm { i d } ) _ { \# } \Gamma \in \mathcal { P } ( \mathcal { P } _ { 2 } ( M ) \times \Gamma _ { T } ) .
$$

Its first marginal is $( e _ { 0 } ) _ { \# } \Gamma = \mathbb { P } _ { 0 }$ , and its second marginal is Γ. Since $\mathcal { P } _ { 2 } ( \mathcal { M } )$ and $\Gamma _ { T }$ are Polish spaces, disintegration with respect to the first marginal gives a measurable family of conditional path measures $\{ \Gamma _ { \mu } \} _ { \mu \in \mathscr { P } _ { 2 } ( \mathcal { M } ) }$ such that

$$
\widehat { \Gamma } = \int _ { { \mathcal P } _ { 2 } ( { \mathcal M } ) } \delta _ { \mu } \otimes \Gamma _ { \mu } d \mathbb P _ { 0 } ( \mu ) .
$$

Here $\Gamma _ { \mu }$ is the conditional law of the trajectory given its initial value $\mu .$ . In particular, for P<sub>0</sub>-almost every $\mu ,$

$$
\Gamma _ { \mu } \big ( \{ \gamma \in \Gamma _ { T } : \gamma ( 0 ) = \mu \} \big ) = 1 .
$$

The concentration properties of Γ also hold under $\Gamma _ { \mu }$ for $\mathbb { P } _ { 0 } { \mathrm { - a l m o s t } }$ every $\mu { : }$ its trajectories are $W _ { 2 }$ -absolutely continuous, remain in ${ \mathcal { P } } ( K )$ , and solve (Measure CE).

Under the additional uniqueness assumption, there is at most one such trajectory starting from $\mu .$ . Since $\Gamma _ { \mu }$ is a probability measure concentrated on these trajectories,

there is exactly one, denoted $\gamma _ { \mu } .$ , and

$$
\Gamma _ { \mu } = \delta _ { \gamma _ { \mu } } \qquad \mathrm { f o r } \ \mathbb { P } _ { 0 } \mathrm { - a l m o s t \ e v e r y } \ \mu .
$$

The measurability of the conditional measures therefore gives a measurable trajectory map $\mu \mapsto \gamma _ { \mu }$ up to null sets. Define $\Phi _ { t } ( \mu ) : = \gamma _ { \mu } ( t )$ , so that $\Phi _ { 0 } ( \mu ) = \mu$ for $\mathbb { P } _ { 0 } .$ -almost every $\mu .$

To identify the law at time t, take any bounded Borel functional $\varphi : \mathscr { P } _ { 2 } ( \mathcal { M } ) \to \mathbb { R }$ The marginal identity and disintegration give

$$
\begin{array} { l } { \displaystyle \int _ { \mathcal P _ { 2 } ( M ) } \varphi ( \boldsymbol \nu ) d \mathbb P _ { t } ( \boldsymbol \nu ) = \int _ { \Gamma _ { T } } \varphi ( \boldsymbol \gamma ( t ) ) d \Gamma ( \boldsymbol \gamma ) } \\ { = \int _ { \mathcal P _ { 2 } ( M ) } \left( \int _ { \Gamma _ { T } } \varphi ( \boldsymbol \gamma ( t ) ) d \Gamma _ { \mu } ( \boldsymbol \gamma ) \right) d \mathbb P _ { 0 } ( \mu ) } \\ { = \int _ { \mathcal P _ { 2 } ( M ) } \varphi ( \boldsymbol \gamma _ { \mu } ( t ) ) d \mathbb P _ { 0 } ( \mu ) } \\ { = \int _ { \mathcal P _ { 2 } ( M ) } \varphi ( \Phi _ { t } ( \mu ) ) d \mathbb P _ { 0 } ( \mu ) . } \end{array}
$$

This is precisely the pushforward identity

$$
\mathbb { P } _ { t } = ( \Phi _ { t } ) _ { \# } \mathbb { P } _ { 0 } , \qquad t \in [ 0 , T ] ,
$$

which establishes the deterministic Lagrangian representation.

C.1.3. Example: Flowing between two diracs centered at $\mu _ { 0 }$ and $\mu _ { 1 }$ , with $\mathcal { M } = \mathbb { R } ^ { d }$ . We consider $\mathrm { a }$ probability path in $\mathcal { P } _ { 2 } ( \mathcal { P } _ { 2 } ( \mathcal { M } ) )$ between two dirac distributions, centered at $\mu _ { 0 }$ and $\mu _ { 1 }$ . In that case, we have $\mathbb { P } _ { t } = \delta _ { \mu _ { t } }$ . We first consider the case where the underlying manifold is the Euclidean space. We treat the general case where M is an arbitrary Riemannian manifold in the next section.

We take the path $\gamma$ to be the constant-speed geodesics connecting $\mu _ { 0 }$ and $\mu _ { 1 }$

$$
\gamma ( t ) = \mu _ { t } = ( \psi _ { t } ) _ { \# } \mu _ { 0 }
$$

$$
\psi _ { t } ( x ) : = ( 1 - t ) x + t T _ { 0 } ( x ) , \qquad t \in [ 0 , 1 ] ,
$$

and $T _ { 0 }$ the optimal transport map on $\mathbb { R } ^ { d }$ between $\mu _ { 0 }$ and $\mu _ { 1 }$ .

The minimal norm tangent vector is the vector field $v _ { t } \in T _ { \mu _ { t } } \mathscr { P } _ { 2 } ( \mathcal { M } )$

$$
v _ { t } ( z ) = \partial _ { t } \psi _ { t } ( x ) | _ { x = S _ { t } ( z ) } = T ( S _ { t } ( z ) ) - S _ { t } ( z ) = ( T - I d ) ( \psi _ { t } ^ { - 1 } ( z ) ) ,
$$

with $S _ { t } ( x ) = \psi _ { t } ^ { - 1 }$

We now need to define a vector $V _ { t }$ on $\mathcal { P } _ { 2 } ( \mathcal { M } )$ that generates this path:

$$
V _ { t } ( \mu ) = { \left\{ \begin{array} { l l } { v _ { t } \quad { \mathrm { i f ~ } } \mu = \mu _ { t } } \\ { 0 \quad { \mathrm { o t h e r w i s e } } } \end{array} \right. }
$$

We now show that $( \delta _ { \mu _ { t } } , V _ { t } )$ solves the continuity equation on $\mathcal { P } _ { 2 } ( \mathcal { M } )$ . We first show that $\left( \mu _ { t } , v _ { t } \right)$ follows the continuity equation on $\mathbb { R } ^ { d }$ , then lift it to show that it follows the continuity equation on $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ .

$( \mu _ { t } , v _ { t } )$ on $\mathbb { R } ^ { d }$ . From the definition of $\mu _ { t }$ and the change of variable formula, we have that

$$
\int _ { \mathbb { R } ^ { d } } \xi ( z ) d \mu _ { t } ( z ) = \int _ { \mathbb { R } ^ { d } } \xi ( \psi _ { t } ( x ) ) d \mu _ { 0 } ( x )
$$

for any continuous test function ξ. Diferentiating with respect to t (and noting that $\dot { \psi } _ { t } = T ( x ) - x )$ , we obtain

$$
\begin{array} { r l } & { \cfrac { d } { d t } \displaystyle \int _ { \mathbb { R } ^ { d } } \xi ( z ) d \mu _ { t } ( z ) = \int _ { \mathbb { R } ^ { d } } \nabla \xi ( \psi _ { t } ( x ) ) \cdot ( T ( x ) - x ) d \mu _ { 0 } ( x ) } \\ & { \qquad = \displaystyle \int _ { \mathbb { R } ^ { d } } \nabla \xi ( z ) \cdot ( T - I d ) ( \psi _ { t } ^ { - 1 } ( z ) ) d \mu _ { t } ( z ) = \int _ { \mathbb { R } ^ { d } } \nabla \xi ( z ) \cdot v _ { t } ( z ) d \mu _ { t } ( z ) } \end{array}
$$

which is the weak formulation of the continuity equation on $\mathbb { R } ^ { d }$

We can now adapt it to time cylindrical functions of the form

$$
\varphi _ { t } ( \mu ) = F \Big ( t , \int _ { \mathbb { R } ^ { d } } \phi _ { 1 } d \mu , \dots , \int _ { \mathbb { R } ^ { d } } \phi _ { k } d \mu \Big ) .
$$

Using the identity above, we have

$$
\frac { d } { d t } \int _ { \mathbb { R } ^ { d } } \phi _ { i } ( z ) d \mu _ { t } ( z ) = \int _ { \mathbb { R } ^ { d } } \nabla \phi _ { i } ( z ) \cdot \boldsymbol { v } _ { t } ( z ) d \mu _ { t } ( z )
$$

We thus have

$$
\frac { d } { d t } \varphi _ { t } ( \mu _ { t } ) = \partial _ { t } { \cal F } ( t , G ( t ) ) + \sum _ { i = 1 } ^ { k } \partial _ { x _ { i } } { \cal F } ( t , \int _ { { \mathbb R } ^ { d } } \phi _ { 1 } d \mu , \ldots , \int _ { { \mathbb R } ^ { d } } \phi _ { k } d \mu ) \cdot \int _ { { \mathbb R } ^ { d } } \nabla \phi _ { i } ( z ) \cdot v _ { t } ( z ) d \mu _ { t } ( z )
$$

Using the definition of the Wasserstein gradient of the cylinder function, we write

$$
\begin{array} { r } { \displaystyle \frac { d } { d t } \varphi _ { t } ( \mu _ { t } ) = \partial _ { t } \varphi _ { t } ( \mu _ { t } ) + \int _ { \mathbb R ^ { d } } \nabla _ { W } \varphi _ { t } ( \mu _ { t } ) \cdot v _ { t } d \mu _ { t } } \\ { = \partial _ { t } \varphi _ { t } ( \mu _ { t } ) + \langle \nabla _ { W } \varphi _ { t } ( \mu _ { t } ) , v _ { t } \rangle _ { L ^ { 2 } ( \mu _ { t } ) } } \end{array}
$$

which is the our desired results on $\mathbb { R } ^ { d }$ that we now need to lift to $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ $( \mathbb { P } _ { t } , V _ { t } )$ on $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ . We have

$$
\begin{array} { l l } { \displaystyle \int _ { 0 } ^ { 1 } \int _ { \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) } \left( \partial _ { t } \varphi _ { t } ( \mu ) + \langle \nabla _ { \mathcal { W } } \varphi _ { t } ( \mu ) , V _ { t } ( \mu ) \rangle _ { L ^ { 2 } ( \mu ) } \right) d \mathbb { P } _ { t } ( \mu ) d t = } \\ { \displaystyle \int _ { 0 } ^ { 1 } \left( \partial _ { t } \varphi _ { t } ( \mu _ { t } ) + \langle \nabla _ { \mathcal { W } } \varphi _ { t } ( \mu _ { t } ) , V _ { t } ( \mu _ { t } ) \rangle _ { L ^ { 2 } ( \mu _ { t } ) } \right) d t } \end{array}
$$

which is equal to $\begin{array} { r } { \int _ { 0 } ^ { 1 } \frac { d } { d t } \varphi _ { t } ( \mu _ { t } ) d t = \varphi _ { 1 } ( \mu _ { 1 } ) - \varphi _ { 0 } ( \mu _ { 0 } ) } \end{array}$ and vanishes with the assumption that $\varphi _ { t }$ has a compact support in time. Hence, we have

$$
\int _ { 0 } ^ { 1 } \int _ { \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } ) } \big ( \partial _ { t } \varphi _ { t } ( \mu ) + \langle \nabla _ { \mathcal { W } } \varphi _ { t } ( \mu ) , V _ { t } ( \mu ) \rangle _ { L ^ { 2 } ( \mu ) } \big ) \ d \mathbb { P } _ { t } ( \mu ) d t = 0 ,
$$

which shows that $( \mathbb { P } _ { t } , V _ { t } )$ solves the weak continuity equation on $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$

C.1.4. Example: Flowing between two diracs on arbitrary Riemannian manifolds. We now generalize the example above to general Riemannian manifolds. Considering arbitrary manifolds $( \mathcal { M } , g )$ , and writing geodesics between x and $y \in \mathcal { M }$

$$
\alpha _ { t } ( x , y ) : = \exp _ { x } ( t \log _ { x } y ) , \qquad t \in [ 0 , 1 ] ,
$$

where,

$\exp _ { x } : T _ { x } M  M$ is the exponential map,

• log<sub>x</sub> $y \in T _ { x } M$ is the inverse $( \exp _ { x } ( \log _ { x } y ) = y )$

The McCann displacement interpolation on M is defined as

$$
\psi _ { t } ( x ) : = \alpha _ { t } ( x , T ( x ) ) = \exp _ { x } ( t \log _ { x } T ( x ) ) , \qquad \mu _ { t } : = ( \psi _ { t } ) _ { \# } \mu _ { 0 } ,
$$

which is the Riemannian analogue of the Euclidean straight-line interpolation and is still a constant-speed W<sub>2</sub>-geodesic on $\mathcal { P } _ { 2 } ( M )$

The velocity field then writes

$$
v _ { t } ( \cdot | \mu _ { 0 } , \mu _ { 1 } ) = \frac { d } { d t } \psi _ { t } ( x ) .
$$

We define $\mathbb { P } _ { t } = \delta _ { \mu _ { t } }$ and $V _ { t } ( \mu ) = v _ { t } { \mathrm { ~ i f ~ } } \mu = \mu _ { t }$ and 0 otherwise.

We can now follow the same procedure as in the Euclidean case to show that $( V _ { t } , \mathbb { P } _ { t } )$ solves the weak continuity equation on $\mathcal { P } _ { 2 } ( \mathcal { M } )$ . We first verify it on M and then lift it to $\mathcal { P } _ { 2 } ( \mathcal { M } )$

$( \mu _ { t } , v _ { t } )$ on $\mathcal { M }$ . From the definition of $\mu _ { t }$ and the change of variable formula, we have that

$$
\int _ { \mathcal { M } } \xi ( z ) d \mu _ { t } ( z ) = \int _ { \mathcal { M } } \xi ( \psi _ { t } ( x ) ) d \mu _ { 0 } ( x )
$$

for any continuous test function $\xi .$ Diferentiating with respect to t (and using that $v _ { t } ( \psi _ { t } ( x ) ) = \dot { \psi } _ { t } ( x ) )$ ), we obtain

$$
\begin{array} { r l } { \displaystyle \frac { d } { d t } \int _ { \mathcal { M } } \xi ( z ) d \mu _ { t } ( z ) = \int _ { \mathcal { M } } \langle \nabla \xi ( \psi _ { t } ( x ) ) , \dot { \psi } _ { t } \rangle _ { g } d \mu _ { 0 } ( x ) } & { { } } \\ { = \displaystyle \int _ { \mathcal { M } } \langle \nabla \xi ( \psi _ { t } ( x ) ) , v _ { t } ( \psi _ { t } ( x ) ) \rangle _ { g } d \mu _ { 0 } ( x ) } & { { } } \\ { \displaystyle } & { { } = \displaystyle \int _ { \mathcal { M } } \langle \nabla \xi ( z ) , v _ { t } ( z ) \rangle _ { g } d \mu _ { t } ( z ) } \end{array}
$$

This can again be extended to time-dependent cylinder functions $\varphi _ { t }$ to yield

$$
\begin{array} { c } { \displaystyle \frac { d } { d t } \varphi _ { t } ( \mu _ { t } ) = \partial _ { t } \varphi _ { t } ( \mu _ { t } ) + \int _ { \mathcal { M } } \langle \nabla _ { W } \varphi _ { t } ( \mu _ { t } ) , v _ { t } \rangle _ { g } d \mu _ { t } } \\ { = \partial _ { t } \varphi _ { t } ( \mu _ { t } ) + \langle \nabla _ { W } \varphi _ { t } ( \mu _ { t } ) , v _ { t } \rangle _ { L ^ { 2 } ( \mu _ { t } ) } } \end{array}
$$

where $\left. \cdot , \cdot \right. _ { L ^ { 2 } ( \mu _ { t } ) }$ implicitly encodes the Riemannian metric in the inner product. $( \mathbb { P } _ { t } , V _ { t } )$ on $\mathcal { P } _ { 2 } ( \mathcal { M } )$ . Since $\mathbb { P } _ { t }$ is concentrated on $\mu _ { t }$ , we have:

$$
\begin{array} { l l } { \displaystyle \int _ { 0 } ^ { 1 } \int _ { \mathcal { P } _ { 2 } ( \mathcal { M } ) } \left( \partial _ { t } \varphi _ { t } ( \mu ) + \langle \nabla _ { \mathcal { W } } \varphi _ { t } ( \mu ) , V _ { t } ( \mu ) \rangle _ { L ^ { 2 } ( \mu ) } \right) d \mathbb { P } _ { t } ( \mu ) d t = } \\ { \displaystyle \int _ { 0 } ^ { 1 } \left( \partial _ { t } \varphi _ { t } ( \mu _ { t } ) + \langle \nabla _ { \mathcal { W } } \varphi _ { t } ( \mu _ { t } ) , V _ { t } ( \mu _ { t } ) \rangle _ { L ^ { 2 } ( \mu _ { t } ) } \right) d t } \end{array}
$$

which, from our derivation above, is equal to $\begin{array} { r } { \int _ { 0 } ^ { 1 } \frac { d } { d t } \varphi _ { t } ( \mu _ { t } ) d t = \varphi _ { 1 } ( \mu _ { 1 } ) - \varphi _ { 0 } ( \mu _ { 0 } ) } \end{array}$ and vanishes with the assumption that $\varphi _ { t }$ has a compact support in time. Hence, we have

$$
\int _ { 0 } ^ { 1 } \int _ { \mathcal { P } _ { 2 } ( \mathcal { M } ) } \big ( \partial _ { t } \varphi _ { t } ( \mu ) + \langle \nabla _ { \mathcal { W } } \varphi _ { t } ( \mu ) , V _ { t } ( \mu ) \rangle _ { L ^ { 2 } ( \mu ) } \big ) \ d \mathbb { P } _ { t } ( \mu ) d t = 0 ,
$$

which shows that $( \mathbb { P } _ { t } , V _ { t } )$ solves the weak continuity equation on $\mathcal { P } _ { 2 } ( \mathcal { M } )$

C.2. Marginal vector field generates marginal probability path. In this section, we show that marginalizing conditional velocity fields preserves the weak continuity equation. Under the hypotheses of Theorem C.3 for the marginal pair, including trajectory uniqueness, the marginal vector field also generates the marginal probability path.

Assumption C.4 (Conditional velocity and probability path solve continuity equation). We assume that for any given boundary measures $\mu _ { 0 } , \mu _ { 1 }$ , the conditional flow $\left( \mathbb { P } _ { t } ( \cdot | \mu _ { 0 } , \mu _ { 1 } ) \right)$ <sub>t</sub> and its velocity field $V _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } )$ satisfy the weak continuity equation. This means for any suitable test (cylinder) function $\varphi _ { t } ( \mu )$ , the following holds:

$$
\int _ { 0 } ^ { T } \int _ { \mathcal { P } _ { 2 } ( \mathcal { M } ) } \big ( \partial _ { t } \varphi _ { t } ( \mu ) + \langle \nabla _ { \mathcal { W } } \varphi _ { t } ( \mu ) , V _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) \rangle _ { L ^ { 2 } ( \mu ) } \big ) \ d \mathbb { P } _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) d t = 0 .
$$

Definition C.5 (Marginal velocity field). Analogously to classical flow matching, we define the marginal velocity field $V _ { t } ( \mu )$ as the conditional expectation of the conditional velocity fields $V _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } )$

$$
V _ { t } ( \mu ) : = \mathbb { E } _ { \mathbb { P } _ { t } ( \mu _ { 0 } , \mu _ { 1 } | \mu ) } [ V _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) ] = \int _ { \mathcal { P } _ { 2 } ( \mathcal { M } ) } V _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) \ d \mathbb { P } _ { t } ( \mu _ { 0 } , \mu _ { 1 } | \mu )
$$

where $\begin{array} { r } { d \mathbb { P } _ { t } ( \mu _ { 0 } , \mu _ { 1 } | \mu ) = \frac { d \mathbb { P } _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) d \Pi ( \mu _ { 0 } , \mu _ { 1 } ) } { d \mathbb { P } _ { t } ( \mu ) } } \end{array}$ is the conditional probability measure.

This definition implies the key relation:

$$
V _ { t } ( \mu ) d \mathbb { P } _ { t } ( \mu ) = \int _ { \mathcal { P } _ { 2 } ( \mathcal { M } ) \times \mathcal { P } _ { 2 } ( \mathcal { M } ) } V _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) d \mathbb { P } _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) d \Pi ( \mu _ { 0 } , \mu _ { 1 } )
$$

Theorem C.6 (Marginal vector field generates the marginal probability path). Assume the conditional pair $( \mathbb { P } _ { t } ( \cdot | \mu _ { 0 } , \mu _ { 1 } ) , V _ { t } ( \cdot | \mu _ { 0 } , \mu _ { 1 } ) )$ satisfies Assumption $C . 4 \cdot$ . Then the marginal vector field $V _ { t }$ from Definition C.5 and the marginal probability path $\mathbb { P } _ { t }$ solve (Weak CE). Under the hypotheses of the superposition theorem (Theorem $C . 3 )$ for the marginal pair, including trajectory uniqueness, $V _ { t }$ generates the marginal probability path $\mathbb { P } _ { t }$

Proof of Theorem C.6. We want to prove that the marginal flow $\mathbb { P } _ { t }$ also satisfies the weak continuity equation with an appropriately defined marginal velocity field $V _ { t } ( \mu )$ . That is, we aim to show that:

$$
\int _ { 0 } ^ { T } \int _ { \mathcal { P } _ { 2 } ( \mathcal { M } ) } \big ( \partial _ { t } \varphi _ { t } ( \mu ) + \langle \nabla _ { \mathcal { W } } \varphi _ { t } ( \mu ) , V _ { t } ( \mu ) \rangle _ { L ^ { 2 } ( \mu ) } \big ) \ d \mathbb { P } _ { t } ( \mu ) d t = 0 .
$$

Let’s evaluate the left-hand side of the target equation by substituting the definition of the marginal flow $d \mathbb { P } _ { t } ( \mu )$ . We can split the expression into two parts.

Temporal derivative term. We begin with the term containing the partial derivative with respect to time, $\partial _ { t } \varphi _ { t } ( \mu )$ . We have

$$
\int _ { 0 } ^ { T } \int _ { \mathcal { P } _ { 2 } ( \mathcal { M } ) } \partial _ { t } \varphi _ { t } ( \mu ) ~ d \mathbb { P } _ { t } ( \mu ) d t
$$

1. Substitute the definition of the marginal measure $d \mathbb { P } _ { t } ( \mu )$ :

$$
= \int _ { 0 } ^ { T } \int _ { \mathcal { P } _ { 2 } ( \mathcal { M } ) } \partial _ { t } \varphi _ { t } ( \mu ) \left( \int _ { \mathcal { P } _ { 2 } ( \mathcal { M } ) \times \mathcal { P } _ { 2 } ( \mathcal { M } ) } d \mathbb { P } _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) d \Pi ( \mu _ { 0 } , \mu _ { 1 } ) \right) d t
$$

2. By Fubini’s theorem, we can exchange the order of integration:

$$
= \int _ { { \mathcal { P } } _ { 2 } ( { \mathcal { M } } ) \times { \mathcal { P } } _ { 2 } ( { \mathcal { M } } ) } \left( \int _ { 0 } ^ { T } \int _ { { \mathcal { P } } _ { 2 } ( { \mathcal { M } } ) } \partial _ { t } \varphi _ { t } ( \mu ) \ d { \mathbb { P } } _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) d t \right) d \Pi ( \mu _ { 0 } , \mu _ { 1 } )
$$

3. Using the assumed conditional continuity equation, we replace the inner integral:

$$
= \int _ { { \mathcal { P } } _ { 2 } ( M ) \times { \mathcal { P } } _ { 2 } ( M ) } \biggl ( - \int _ { 0 } ^ { T } \int _ { { \mathcal { P } } _ { 2 } ( M ) } \langle \nabla _ { { \mathcal { W } } } \varphi _ { t } ( \mu ) , V _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) \rangle _ { L ^ { 2 } ( \mu ) }  \\  \qquad \cdot \ d { \mathbb { P } } _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) d t \biggr ) d \Pi ( \mu _ { 0 } , \mu _ { 1 } )
$$

4. Combining the integrals gives our final expression for the first part:

$$
\mathrm { P a r t ~ 1 } = - \int _ { \mathscr { P } _ { 2 } ( M ) \times \mathscr { P } _ { 2 } ( M ) } \int _ { 0 } ^ { T } \int _ { \mathscr { P } _ { 2 } ( M ) } \langle \nabla _ { \mathscr { W } } \varphi _ { t } ( \mu ) , V _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) \rangle _ { L ^ { 2 } ( \mu ) }
$$

Velocity term. Now we analyze the second term involving the marginal velocity field $V _ { t } ( \mu )$

$$
\mathrm { P a r t ~ 2 } = \int _ { 0 } ^ { T } \int _ { \mathcal { P } _ { 2 } ( \mathcal { M } ) } \langle \nabla \mathcal { w } \varphi _ { t } ( \mu ) , V _ { t } ( \mu ) \rangle _ { L ^ { 2 } ( \mu ) } ~ d \mathbb { P } _ { t } ( \mu ) d t
$$

1. Substituting the definition of the marginal velocity field:

$$
\mathrm { P a r t ~ 2 } = \int _ { 0 } ^ { T } \int _ { \mathcal { P } _ { 2 } ( M ) } \left. \nabla _ { \mathcal { W } } \varphi _ { t } ( \mu ) , \int _ { \mathcal { P } _ { 2 } ( M ) \times \mathcal { P } _ { 2 } ( M ) } V _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) \ d \mathbb { P } _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) \right. _ { L ^ { 2 } ( \mu ) }
$$

2. Using the linearity of the inner product and the definition of conditional probability, we can combine the integrals:

$$
= \int _ { 0 } ^ { T } \int _ { \mathcal { P } _ { 2 } ( M ) } \int _ { \mathcal { P } _ { 2 } ( M ) \times \mathcal { P } _ { 2 } ( M ) } \langle \nabla _ { \mathcal { W } } \varphi _ { t } ( \mu ) , V _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) \rangle _ { L ^ { 2 } ( \mu ) } d \mathbb { P } _ { t } ( \mu | \mu _ { 0 } , \mu _ { 1 } ) d \Pi ( \mu _ { 0 } , \mu _ { 1 } ) d \boldsymbol { t }
$$

Hence, Part $2 \ = \ - \mathrm { P a r t } \ 1$ , establishing the weak continuity equation for the marginal pair. Under the stated additional hypotheses, Theorem C.3 then gives $\mathbb { P } _ { t } = ( \Phi _ { t } ) _ { \# } \mathbb { P } _ { 0 }$ , completing the proof. □

We have thus shown that if we define the marginal velocity field as the conditional expectation of the conditional velocity fields, the resulting marginal flow satisfies the weak continuity equation on the Wasserstein manifold.

C.3. Equivalence of conditional and non-conditional flow matching losses. A natural loss for the flow matching objective is

$$
\mathcal { L } _ { F M } = \mathbb { E } _ { t , \mu _ { t } \sim \mathbb { P } _ { t } } [ \| V _ { t } ( \mu _ { t } ) - V _ { t } ^ { \theta } ( \mu _ { t } ) \| _ { L ^ { 2 } ( \mu _ { t } ) } ^ { 2 } ]
$$

where $\begin{array} { r } { \| V _ { t } ( \mu ) \| _ { L ^ { 2 } ( \mu ) } ^ { 2 } = \int _ { \mathcal { M } } \| V _ { t } ( \mu ) ( x ) \| _ { g } ^ { 2 } d \mu ( x ) } \end{array}$

The conditional version would then be

$$
\mathcal { L } _ { C F M } = \mathbb { E } _ { t , \mu _ { 0 } , \mu _ { 1 } \sim \Pi , \mu _ { t } \sim \mathbb { P } _ { t } ( \cdot | \mu _ { 0 } , \mu _ { 1 } ) } [ \| V _ { t } ( \mu _ { t } \mid \mu _ { 0 } , \mu _ { 1 } ) - V _ { t } ^ { \theta } ( \mu _ { t } ) \| _ { L ^ { 2 } ( \mu _ { t } ) } ^ { 2 } ]
$$

Following the original flow matching proof Lipman et al. [2024], we obtain

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathcal { L } _ { C F M } = \mathbb { E } _ { t , \mu _ { 0 } , \mu _ { 1 } \sim \Pi , \mu _ { t } \sim \mathbb { P } _ { t } ( \cdot | \mu _ { 0 } , \mu _ { 1 } ) } \big [ \nabla _ { \theta } \| V _ { t } ( \mu _ { t } \mid \mu _ { 0 } , \mu _ { 1 } ) - V _ { t } ^ { \theta } ( \mu _ { t } ) \| _ { L ^ { 2 } ( \mu _ { t } ) } ^ { 2 } \big ] } \\ & { = \mathbb { E } _ { t , \mu _ { 0 } , \mu _ { 1 } \sim \Pi , \mu _ { t } \sim \mathbb { P } _ { t } ( \cdot | \mu _ { 0 } , \mu _ { 1 } ) } \big [ 2 \langle \nabla _ { \theta } V _ { t } ^ { \theta } ( \mu _ { t } ) , V _ { t } ^ { \theta } ( \mu _ { t } ) \rangle - 2 \langle V _ { t } ( \mu _ { t } \mid \mu _ { 0 } , \mu _ { 1 } ) , \nabla _ { \theta } V _ { t } ^ { \theta } ( \mu _ { t } ) \rangle \big ] } \end{array}
$$

Focusing on the right handside in the expectation, we have

$$
\begin{array} { r l } & { \mathbb { E } _ { t , \mu _ { 0 } , \mu _ { 1 } \sim \Pi , \mu _ { t } \sim \mathbb { P } _ { t } ( \cdot \vert \mu _ { 0 } , \mu _ { 1 } ) } [ \langle V _ { t } ( \mu _ { t } \mid \mu _ { 0 } , \mu _ { 1 } ) , \nabla _ { \theta } V _ { t } ^ { \theta } ( \mu _ { t } ) \rangle ] } \\ & { = \mathbb { E } _ { t , \mu _ { t } \sim \mathbb { P } _ { t } } \left[ \mathbb { E } _ { \mu _ { 0 } , \mu _ { 1 } \sim \mathbb { P } ( \cdot \vert \mu _ { t } ) } [ \langle V _ { t } ( \mu _ { t } \mid \mu _ { 0 } , \mu _ { 1 } ) , \nabla _ { \theta } V _ { t } ^ { \theta } ( \mu _ { t } ) \rangle ] \right] } \\ & { = \mathbb { E } _ { t , \mu _ { t } \sim \mathbb { P } _ { t } } [ \langle \mathbb { E } _ { \mu _ { 0 } , \mu _ { 1 } \sim \mathbb { P } ( \cdot \vert \mu _ { t } ) } [ V _ { t } ( \mu _ { t } \mid \mu _ { 0 } , \mu _ { 1 } ) ] , \nabla _ { \theta } V _ { t } ^ { \theta } ( \mu _ { t } ) \rangle ] } \\ & { = \mathbb { E } _ { t , \mu _ { t } \sim \mathbb { P } _ { t } } [ \langle V _ { t } ( \mu _ { t } ) , \nabla _ { \theta } V _ { t } ^ { \theta } ( \mu _ { t } ) \rangle ] } \end{array}
$$

Hence,

$$
\nabla _ { \theta } \mathcal { L } _ { C F M } = \mathbb { E } _ { t , \mu _ { t } \sim \mathbb { P } _ { t } } [ 2 \langle \nabla _ { \theta } V _ { t } ^ { \theta } ( \mu _ { t } ) , V _ { t } ^ { \theta } ( \mu _ { t } ) \rangle - 2 \langle V _ { t } ( \mu _ { t } ) , \nabla _ { \theta } V _ { t } ^ { \theta } ( \mu _ { t } ) \rangle ]\tag{C.1}
$$

To finish the proof, we derive the gradient of $\mathcal { L } _ { F M }$

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathcal { L } _ { F M } = \mathbb { E } _ { t , \mu _ { t } \sim \mathbb { P } _ { t } } [ \nabla _ { \theta } \| V _ { t } ( \mu _ { t } ) - V _ { t } ^ { \theta } ( \mu _ { t } ) \| _ { L ^ { 2 } ( \mu _ { t } ) } ^ { 2 } ] } \\ & { \qquad = \mathbb { E } _ { t , \mu _ { t } \sim \mathbb { P } _ { t } } [ 2 \langle \nabla _ { \theta } V _ { t } ^ { \theta } ( \mu _ { t } ) , V _ { t } ^ { \theta } ( \mu _ { t } ) \rangle - 2 \langle V _ { t } ( \mu _ { t } ) , \nabla _ { \theta } V _ { t } ^ { \theta } ( \mu _ { t } ) \rangle ] , } \end{array}
$$

which coincides exactly with (C.1).

## Appendix D. Experimental Details and additional results.

Here we lay out the settings and hyperparameters used in our experiments. We discuss how datasets were processed, errors and metrics were computed, and model architectures and training procedures.

D.1. Neural Network Architecture & Training. We parameterize the velocity field as a neural network v $_ { \theta } ( X _ { t } , t , c ) : \mathcal { M } ^ { N } \times [ 0 , 1 ] \times \mathcal { C } \to \bar { T } _ { X _ { t } } \mathcal { M } ^ { N }$ , which maps a point cloud $X _ { t } = \{ x _ { 1 } , \ldots , x _ { N } \}$ with $x _ { i } \in { \mathcal { M } }$ , time $t \in [ 0 , 1 ]$ , and optional condition $c \in { \mathcal { C } }$ to a set of tangent vectors at $X _ { t }$ . The network architecture is designed to respect the permutation equivariance of point clouds and optimal transport maps.

The architecture consists of three components: (1) an embedding layer that maps each point $x _ { i } \in { \mathcal { M } }$ to a latent representation, (2) six multi-head self-attention blocks that process the embedded point cloud, and (3) an unembedding layer that projects back to the tangent space $T _ { X _ { t } } { \mathcal { M } } ^ { N }$ . Each self-attention block processes the latent representation $\breve { h } \in \mathbb { R } ^ { \tilde { N } \times d }$ as follows. First, we incorporate temporal and conditional information by adding learned embeddings to form $\tilde { h } = h + \mathrm { E m b e d } _ { t } ( \phi ( t ) ) + \mathrm { E m b e d } _ { c } ( c )$ where $\phi ( t )$ denotes Fourier features of time t. The block then applies:

$$
\begin{array} { r l } & { h ^ { \prime }  h + \mathrm { M H A } ( \mathrm { L a y e r N o r m } ( \tilde { h } ) ) } \\ & { h  h ^ { \prime } + \mathrm { M L P } ( \mathrm { L a y e r N o r m } ( h ^ { \prime } ) ) } \end{array}
$$

where the multi-head attention (MHA) uses 4 heads, and the residual connections are to the original stream h. After the final attention block, the unembedding layer projects the latent representation back to the ambient dimension of M to produce the tangent velocity vectors.

Classifier-Free Guidance. For conditional generation, we employ classifier-free guidance to improve sample quality and controllability. The conditioning embedding $\mathrm { E m b e d } _ { c } ( c )$ projects the condition c to the unit sphere in the embedding dimension, which creates a meaningful semantic separation between null and real conditions. Real conditions lie on the unit sphere while the null condition is represented by the zero vector at the origin. During training, we randomly drop conditions with a probability of $p ,$ and for these null conditions, we set the embedding Embed<sub>c</sub>(c) to the zero vector (not the input c itself). At inference time, we use the classifier-free guidance formula:

$$
v _ { \theta } ^ { \mathrm { C F G } } ( X _ { t } , t , c ) = v _ { \theta } ( X _ { t } , t , \emptyset ) + w \cdot ( v _ { \theta } ( X _ { t } , t , c ) - v _ { \theta } ( X _ { t } , t , \emptyset ) )
$$

where w $\geq 1$ is the guidance weight, and $\varnothing$ denotes the null condition. By default we set $p = 0 . 1$ and $w = 2$

Training Details. By default, we train all models for 500,000 steps using the Adam optimizer [Kingma and Ba, 2014] with a learning rate of $3 \times 1 0 ^ { - 4 }$ . We apply learning rate decay with a factor of 0.99 every 5,000 steps. During training, we sample point clouds of min(N, 1024) (where N is the number of particles in the empirical measure) points from each distribution with a batch size of 32. The time variable t is sampled uniformly from [0, 1] for each training step, and generation is performed using integration via 1000 Euler steps.

For computing the Riemannian entropic map, we set the entropic regularization parameter $\varepsilon = 0 . 0 0 2$ , where all distance matrices are scaled to have a maximum value of 1. The number of Sinkhorn iterations is determined automatically before training by sampling 100 pairs of distributions and finding the minimum number of iterations required to achieve convergence for 95% of the pairs.

Mini-batch OT Coupling. For unconditional generation, we enhance training by matching samples within mini-batches using optimal transport [Pooladian et al., 2023]. This approach efectively performs OT in the Wasserstein space itself [Emami and Pass, 2025, Bonet et al., 2025]. Given source distributions $\{ \bar { \mu } _ { i } \} _ { i = 1 } ^ { B _ { s } } \sim \mathbb { P } _ { 0 }$ (noise) and target distributions $\{ \nu _ { j } \} _ { j = 1 } ^ { B _ { t } } \sim \mathbb { P } _ { 1 }$ (data) within a mini-batch, we construct a cost matrix $C \in \mathbb { R } ^ { B _ { s } \times B _ { t } }$ and sample training pairs $( \mu _ { i } , \nu _ { j } ) \sim \pi ^ { * }$ from the resulting optima coupling $\pi ^ { * }$

To ensure scalability, we compute the cost matrix using the geometric Chamfer distance rather than the Wasserstein distance:

$$
C _ { i , j } = \mathrm { C D } ( \mu _ { i } , \nu _ { j } ) = \frac { 1 } { \left| \mu _ { i } \right| } \sum _ { x \in \mu _ { i } } \operatorname* { m i n } _ { y \in \nu _ { j } } d _ { { \mathcal M } } ( x , y ) + \frac { 1 } { \left| \nu _ { j } \right| } \sum _ { y \in \nu _ { j } } \operatorname* { m i n } _ { x \in \mu _ { i } } d _ { { \mathcal M } } ( x , y )
$$

where $d _ { \mathcal { M } }$ is the intrinsic geodesic distance on the manifold M. The optimal transport plan $\pi ^ { * }$ is computed using the Sinkhorn algorithm with entropic regularization $\varepsilon =$ 0.001, where distance matrices are scaled to maximum value 1 and the number of iterations is determined automatically as described above.

All models were implemented in JAX [Frostig et al., 2019] using optimal transport tools from [Cuturi et al., 2022] and trained on a single NVIDIA B200 GPU, taking 3-6 hours depending on the dataset and manifold geometry.

Source Noise Generation. While rigorously defined distributions exist on non-Euclidean domains (e.g., von Mises or wrapped normal distributions), flow matching only requires a source that is easy to sample from, without needing a closed-form likelihood. We note that naively sampling noise as $\mathrm { p r o j } _ { \mathcal { M } } ( \mathcal { N } ( 0 , I ) )$ (where $\mathrm { p r o j } _ { \mathcal { M } }$ is the projection onto the manifold) fails to produce meaningful variability. Our source distribution must be a distribution over distributions on $\mathcal { M }$ . Indeed, for large enough sample sizes $n ,$ all generated “noise” point clouds become nearly identical, resulting in a degenerate distribution.

Instead, following Haviv et al. [2024], we employ a two-level hierarchical sampling scheme. First, we randomly sample a mean $\mu$ and covariance $\Sigma ,$ then generate points from $\pi ( \mathcal { N } ( \boldsymbol { \mu } , \boldsymbol { \Sigma } ) )$ . The distribution for $\mu$ is derived from the empirical mean and covariance of all points in the training data, while the distribution for Σ is based on the per-sample covariances. Specifically, let $\{ L _ { i } \}$ be the Cholesky factors of the per-sample covariances. We compute the element-wise mean $\bar { L }$ and standard deviation $\sigma _ { L }$ of these lower-triangular matrices, sample a random lower-triangular matrix $\tilde { L } \sim \mathcal { N } ( \bar { L } , \mathrm { d i a g } ( \sigma _ { L } ^ { 2 } ) )$ ), and set $\Sigma = \tilde { L } \tilde { L } ^ { \top }$ . This construction ensures Σ is positive semidefinite. Samples from the resulting mean and covariance are then projected onto the manifold geometry. For high-dimensional datasets $( \mathrm { e . g . }$ , single-cell data on $\mathbb { S } ^ { 1 2 8 - 1 } )$ , we only sample diagonal covariances for computational eficiency.

D.2. Geometric Operations on Manifolds. We provide explicit formulas for the geometric operations used in our method across diferent manifolds. For each geometry, we define: (1) the squared geodesic distance $d _ { \mathcal { M } } ^ { 2 } ( p , q )$ , (2) the geodesic interpolant $\gamma ( t ; p _ { 0 } , p _ { 1 } )$ connecting $p _ { 0 }$ to $p _ { 1 }$ at time $t \in [ 0 , 1 ] , ( 3 )$ the tangent velocity $\begin{array} { r } { v ( t ; p _ { 0 } , p _ { 1 } ) = \frac { d } { d t } \gamma ( t ; p _ { 0 } , p _ { 1 } ) } \end{array}$ , (4) the exponential map $\exp _ { p } ( v , \Delta t )$ that moves from point $p$ along tangent vector v by time $\Delta t .$ , and (5) the tangent norm $\| \cdot \| _ { T _ { p } , M } ^ { 2 }$ used for computing the training loss between predicted and target velocities.

Important note on representation: We always represent manifolds using extrinsic coordinates embedded in Euclidean space $( \mathrm { e . g . , \mathbb { S } ^ { 2 } }$ as 3D unit vectors in $\mathbb { R } ^ { 3 } , \mathbb { H } ^ { 2 }$ as 3D vectors in Lorentz space). This extrinsic representation significantly simplifies neural network modeling, as the network can operate on fixed-dimensional Euclidean vectors while geometric constraints are enforced through projection operations. These formulas correspond directly to our implementation and are included here for completeness and reproducibility.

Euclidean Space R<sup>d</sup>. The standard flat geometry. Points $\boldsymbol { p } \in \mathbb { R } ^ { d }$ have trivial tangent spaces $\bar { T } _ { p } \mathbb { R } ^ { d } \cong \mathbb { R } ^ { d }$ . The geodesics are straight lines. Distance and interpolation:

$$
d _ { \mathbb { R } ^ { d } } ^ { 2 } ( p , q ) = \| p - q \| _ { 2 } ^ { 2 } , \qquad \gamma ( t ; p _ { 0 } , p _ { 1 } ) = ( 1 - t ) p _ { 0 } + t p _ { 1 }
$$

Velocity and exponential map:

$$
v ( t ; p _ { 0 } , p _ { 1 } ) = p _ { 1 } - p _ { 0 } , \qquad \exp _ { p } ( v , \Delta t ) = p + v \cdot \Delta t
$$

Tangent space: $\Vert v - w \Vert _ { T _ { n } \mathbb { R } ^ { d } } ^ { 2 } = \Vert v - w \Vert ^ { 2 }$ . Projection: $\operatorname { p r o j } ( p ) = p .$

d-Torus $\mathbb { T } ^ { d }$ . The d-dimensional torus $\mathbb { T } ^ { d } = ( \mathbb { S } ^ { 1 } ) ^ { d }$ . Points are represented as d angles $p \in [ 0 , 2 \pi ) ^ { d }$ . The geodesic distance accounts for periodic wraparound in each coordinate.

Distance and logarithmic map:

$$
d _ { \mathbb { T } ^ { d } } ^ { 2 } ( p , q ) = \sum _ { i = 1 } ^ { d } \operatorname* { m i n } ( | p _ { i } - q _ { i } | , 2 \pi - | p _ { i } - q _ { i } | ) ^ { 2 } ,
$$

$$
\log _ { p _ { 0 } } ( p _ { 1 } ) = \arctan 2 ( \sin ( p _ { 1 } - p _ { 0 } ) , \cos ( p _ { 1 } - p _ { 0 } ) )
$$

Interpolation and velocity:

$$
\gamma ( t ; p _ { 0 } , p _ { 1 } ) = ( p _ { 0 } + t \cdot \log _ { p _ { 0 } } ( p _ { 1 } ) ) \mod 2 \pi , \qquad v ( t ; p _ { 0 } , p _ { 1 } ) = \log _ { p _ { 0 } } ( p _ { 1 } )
$$

Exponential map: exp $\mathbf { \sigma } _ { p } ( v , \Delta t ) = ( p + v \cdot \Delta t )$ mod $2 \pi$ .

Tangent space: $\Vert v - \dot { w } \Vert _ { T _ { n } \mathbb { T } ^ { d } } ^ { 2 } = \Vert v - w \Vert ^ { 2 }$ . Projection: $\mathrm { p r o j } ( p ) = p$ mod $2 \pi$ .

d-Sphere $\mathbb { S } ^ { d }$ . The d-dimensional sphere embedded in $\mathbb { R } ^ { d + 1 }$ . Points are unit vectors $p \in \mathbb { R } ^ { d + 1 }$ with $\| p \| = 1$ . The tangent space at $p$ is $T _ { p } \mathbb { S } ^ { d } = \left\{ v \in \mathbb { R } ^ { d + 1 } : \langle v , p \rangle = 0 \right\}$ . We use spherical linear interpolation (SLERP).

Distance: $d _ { \mathbb { S } ^ { d } } ^ { 2 } ( p , q ) = \operatorname { a r c c o s } ( \operatorname { c l i p } ( \langle p , q \rangle , - 1 , 1 ) ) ^ { 2 }$ , where $\theta = \operatorname { a r c c o s } ( \langle p _ { 0 } , p _ { 1 } \rangle )$

$$
\gamma ( t ; p _ { 0 } , p _ { 1 } ) = \left\{ \begin{array} { l l } { \frac { \sin ( ( 1 - t ) \theta ) } { \sin ( \theta ) } p _ { 0 } + \frac { \sin ( t \theta ) } { \sin ( \theta ) } p _ { 1 } } & { \mathrm { i f ~ } \sin ( \theta ) \geq 1 0 ^ { - 6 } } \\ { ( 1 - t ) p _ { 0 } + t p _ { 1 } } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

Velocity:

$$
v ( t ; p _ { 0 } , p _ { 1 } ) = { \left\{ \begin{array} { l l } { - { \frac { \theta \cos ( ( 1 - t ) \theta ) } { \sin ( \theta ) } } p _ { 0 } + { \frac { \theta \cos ( t \theta ) } { \sin ( \theta ) } } p _ { 1 } } & { { \mathrm { i f ~ } } \sin ( \theta ) \geq 1 0 ^ { - 6 } } \\ { - p _ { 0 } + p _ { 1 } } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }
$$

Exponential map:

$$
\exp _ { p } ( v , \Delta t ) = \left\{ \begin{array} { l l } { \mathrm { c o s } ( \| v \| \Delta t ) p + \sin ( \| v \| \Delta t ) \frac { v } { \| v \| } } & { \mathrm { i f ~ } \| v \| \ge 1 0 ^ { - 6 } } \\ { \mathrm { n o r m a l i z e } ( p + v \Delta t ) } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

Tangent space: $\lVert \boldsymbol { v } - \boldsymbol { w } \rVert _ { T _ { v } \mathbb { S } ^ { d } } ^ { 2 } = \lVert \boldsymbol { v } _ { \mathrm { t a n } } - \boldsymbol { w } _ { \mathrm { t a n } } \rVert ^ { 2 }$ where $v _ { \mathrm { t a n } } = v - \langle v , p \rangle p$ . Projection: pro $| ( p ) = p / \| p \|$

Hyperbolic Space $\mathbb { H } ^ { d }$ (Lorentz Model). Hyperbolic space in the Lorentz (hyperboloid) model. Points lie on $\{ x \in \mathbb { R } ^ { d + 1 } : \langle x , x \rangle _ { L } = - 1 , x _ { 0 } > 0 \}$ with Minkowski inner product $\langle x , y \rangle _ { L } = - x _ { 0 } y _ { 0 } + \textstyle \sum _ { i = 1 } ^ { d } x _ { i } y _ { i }$ . The tangent space at p is $T _ { p } \mathbb { H } ^ { d } = \{ v : \langle v , p \rangle _ { L } =$ 0}.

Distance:

$$
\begin{array} { r } { d _ { \mathbb { H } ^ { d } } ^ { 2 } ( p , q ) = \operatorname { a r c c o s h } \bigl ( \operatorname { c l i p } ( - \langle p , q \rangle _ { L } , 1 + 1 0 ^ { - 7 } , \infty ) \bigr ) ^ { 2 } , \quad \omega = \operatorname { a r c c o s h } ( - \langle p _ { 0 } , p _ { 1 } \rangle _ { L } ) . } \end{array}
$$

Interpolation:

$$
\gamma ( t ; p _ { 0 } , p _ { 1 } ) = \left\{ \begin{array} { l l } { \frac { \sinh ( ( 1 - t ) \omega ) } { \sinh ( \omega ) } p _ { 0 } + \frac { \sinh ( t \omega ) } { \sinh ( \omega ) } p _ { 1 } } & { \mathrm { i f ~ \sinh ( \omega ) \geq 1 0 ^ { - 6 } ~ } } \\ { ( 1 - t ) p _ { 0 } + t p _ { 1 } } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

Velocity:

$$
v ( t ; p _ { 0 } , p _ { 1 } ) = { \left\{ \begin{array} { l l } { { \frac { - \omega \cosh ( ( 1 - t ) \omega ) } { \sinh ( \omega ) } } p _ { 0 } + { \frac { \omega \cosh ( t \omega ) } { \sinh ( \omega ) } } p _ { 1 } } & { \sinh ( \omega ) \geq 1 0 ^ { - 6 } } \\ { - p _ { 0 } + p _ { 1 } } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }
$$

Exponential map (with $c = \| v \| _ { L } , \ \| v \| _ { L } = \sqrt { \operatorname* { m a x } ( \langle v , v \rangle _ { L } , 0 ) } )$

$$
\exp _ { p } ( v , \Delta t ) = \left\{ \begin{array} { l l } { \cosh ( c \Delta t ) p + \frac { \sinh ( c \Delta t ) } { c } v } & { c \geq 1 0 ^ { - 6 } } \\ { \mathrm { p r o j } ( p + v \Delta t ) } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

Tangent space: $\lVert \boldsymbol { v } - \boldsymbol { w } \rVert _ { T _ { v } \mathbb { H } ^ { d } } ^ { 2 } = \langle \boldsymbol { v } _ { \mathrm { t a n } } - \boldsymbol { w } _ { \mathrm { t a n } } , \boldsymbol { v } _ { \mathrm { t a n } } - \boldsymbol { w } _ { \mathrm { t a n } } \rangle _ { L }$ where $\begin{array} { r } { v _ { \mathrm { t a n } } = v + \langle v , p \rangle _ { L } p , } \end{array}$ Projection: pro $\mathfrak { j } ( p ) = [ \sqrt { 1 + \| p _ { 1 : d } \| ^ { 2 } } , p _ { 1 } , \dotsc , p _ { d } ] ^ { \top }$

Handling cut loci on the manifolds.. The validity of our theorems relies on the assumption that the log map is single valued and smooth. That said, for the manifolds considered above, the cut locus of a fixed point is a measure-zero set, such that ambiguity occurs with probability zero. In practice, our implementation numerically disambiguates these situations by returning a single value for the logarithmic map, as the above velocity expressions show.

D.3. Benchmarking metrics for generation of distributions on Manifolds. For unconditional generation, we compare real and generated distributions using a classwise nearest-neighbor accuracy deviation and Maximum Mean Discrepancy (MMD), building on point-cloud evaluation measures such as those in Zhou et al. [2021]. Each observation in this evaluation is an entire point cloud. We generate as many clouds as there are clouds in the held-out test set, pool the two sets, and classify each cloud as real or generated using the label of its nearest neighbor, excluding the cloud itself. Distances between clouds are computed using geometric CD or EMD.

<table><tr><td>Dataset</td><td>Method</td><td>Train N</td><td>Test N 1,010</td><td>|P| Sink. iters</td><td>Time 2h 15m</td></tr><tr><td>MNIST 3</td><td>FM RFM Set-FM Set-RFM WFM RWEFM PSF</td><td>6,131</td><td></td><td>237 1790 850</td><td>1h 39m 4h 17m 3h 08m 9h 12m 6h 04m ~12h ~12h</td></tr><tr><td>MNIST 4</td><td>PVD FM RFM Set-FM Set-RFM WFM RWEFM PSF PVD</td><td>5,842</td><td>982</td><td>215 2400 930</td><td>2h11m 1h 36m 4h 13m 3h11m 10h 54m 6h 00m ~12h ~12h</td></tr><tr><td>MNIST 8</td><td>FM RFM Set-FM Set-RFM WFM RWEFM PSF PVD</td><td>5,851</td><td>974</td><td>276 1710 810</td><td>2h 12m 1h41m 4h 02m 3h12m 10h 29m 6h 34m ~12h ~12h</td></tr><tr><td>EMNIST h</td><td>FM RFM Set-FM Set-RFM WFM RWEFM PSF PVD</td><td>4,800</td><td>800</td><td>300 490 600</td><td>46m 46m 2h 03m 1h 58m 3h 33m 4h 06m ~12h ~12h</td></tr><tr><td>EMNIST w</td><td>FM RFM Set-FM Set-RFM WFM RWEFM PSF PVD FM</td><td>4,800 4,800</td><td>800</td><td>306 530 720</td><td>46m 47m 2h 01m 2h 00m 3h 52m 4h 22m ~12h ~12h 43m</td></tr><tr><td>EMNIST y KMNIST ki</td><td>RFM Set-FM Set-RFM WFM RWEFM PSF PVD</td><td></td><td>800</td><td>278 540 650</td><td>44m 2h 09m 1h 50m 3h 42m 4h 00m ~12h ~12h 1h 30m</td></tr><tr><td></td><td>FM RFM Set-FM Set-RFM WFM RWEFM FM</td><td>6,000 6,000</td><td>1,000</td><td>464 420 440</td><td>1h 44m 2h 13m 2h 15m 4h 51m 5h11m 1h51m 2h 03m</td></tr><tr><td>KMNIST na KMNIST ma</td><td>RFM Set-FM Set-RFM WFM RWEFM FM</td><td>6,000</td><td>1,000</td><td>418 350 480</td><td>2h 43m 2h 53m 4h 23m 5h 09m 1h 47m</td></tr><tr><td></td><td>RFM Set-FM Set-RFM WFM RWEFM Set-FM</td><td>3,432</td><td>1,000 858</td><td>428 420 460</td><td>1h37m 2h 13m 2h41m 4h 37m 5h 05m 4h 23m</td></tr><tr><td>MDcath</td><td>Set-RFM WFM RWEFM</td><td></td><td></td><td>127,000 600 1090</td><td>5h11m 7h 32m 9h 15m</td></tr><tr><td>scRNA-seq</td><td>Set-FM Set-RFM WFM RWEFM</td><td>1,200</td><td>240</td><td>5,000 420 310</td><td>7h 33m 8h 44m 16h 08m 14h 47m</td></tr></table>

Table S1: Wall-clock training times Set-FM/Set-RFM pay an extra self-attention cost over point-cloud pairs; WFM and RWEFM pay a per-step Sinkhorn OT cost whose iteration count (column 6) is determined automatically per experiment. Even on<sup>46</sup> the same dataset, WFM and RWEFM require diferent iteration counts because they use diferent distance kernels (ambient Euclidean vs. Riemannian geodesic) leading to diferent convergence rates. Dashes (–): no OT coupling.

![](images/719020a4cbdc23abf6c8c781968940f36172758895d5726b2c7d36bf954f7a6c.jpg)  
Fig. S1: Individual whole single-cell samples generated by RWEFM. We show additional individual samples generated by RWEFM in the latent space of SCimilarity [Heimberg et al., 2025], a foundation model for single-cell data that operates in $\mathbb { S } ^ { 1 2 8 - \breve { 1 } }$ Each panel shows the UMAP visualization of a single-cell sample, with cells colored by their cell type. Generated samples (left) closely match the cellular composition and structure of true samples (right).

Let $a _ { \mathrm { r e a l } }$ be the fraction of real clouds correctly classified as real and $a _ { \mathrm { g e n } }$ the fraction of generated clouds correctly classified as generated. We report the classwise 1-NN deviation abbreviated 1-NN-D in the tables. In particular, this is not the raw pooled classification accuracy. With equally sized classes, the pooled accuracy is $( a _ { \mathrm { r e a l } } + a _ { \mathrm { g e n } } ) / 2$ . Both $( a _ { \mathrm { r e a l } } , a _ { \mathrm { g e n } } ) = ( 1 , 0 )$ and (0.5, 0.5) give pooled accuracy 0.5, although the first case labels every cloud as real. Our score distinguishes these cases, giving 0.5 and 0, respectively.

Lower values therefore indicate smaller classwise deviations from 50% accuracy. The minimum value 0 means that both empirical classwise accuracies equal 0.5; the maximum is 0.5. Finite-sample fluctuations can yield a nonzero score even when the real and generated distributions coincide.

We also compute the Maximum Mean Discrepancy (MMD) using both geometric CD and EMD kernels. The MMD is defined as:

$$
\mathrm { M M D } ^ { 2 } ( \mu , \nu ) = \mathbb { E } _ { x , x ^ { \prime } \sim \mu } [ k ( x , x ^ { \prime } ) ] + \mathbb { E } _ { y , y ^ { \prime } \sim \nu } [ k ( y , y ^ { \prime } ) ] - 2 \mathbb { E } _ { x \sim \mu , y \sim \nu } [ k ( x , y ) ]
$$

where $k ( \cdot , \cdot )$ is a kernel function. We use $k ( x , y ) = \exp ( - d ( x , y ) / \sigma )$ where d is either the geometric CD or EMD distance, and $\sigma = 0 . 1$ is a scaling factor. We report both MMD-CD and MMD-EMD values.

For conditional generation, where we have a known ground-truth target distribution we are trying to match, we directly compare the generated distribution to the groundtruth using geometric Wasserstein distances $W _ { 1 }$ and $W _ { 2 } .$ , as well as MMD. All distances are computed using the intrinsic geometry of the underlying manifold.

D.4. MNIST & EMNIST on Sphere and Hyperbolic Space. In Figure 2 we show how RWEFM can learn to generate distributions on the sphere and hyperbolic space. We use the MNIST digit dataset [LeCun et al., 1998] for the sphere experiments and EMNIST letters [Cohen et al., 2017] for the hyperbolic experiments. Both datasets consist of $2 8 \times 2 8$ grayscale images of handwritten digits/letters. We convert each image to a point cloud in $\mathbb { R } ^ { 2 }$ by thresholding each pixel and normalizing the coordinates to [−1, 1]. This transforms each image from a point in $\mathbb { R } ^ { 2 8 \times 2 8 }$ to a distribution over $\mathbb { R } ^ { 2 }$

For MNIST, we convert the point-cloud over $\mathbb { R } ^ { 2 }$ to a distribution over $\mathbb { S } ^ { 2 }$ . We treat the x and y coordinates of each point as longitude and latitude on the sphere and produce the $3 D$ coordinates via the spherical to Cartesian conversion. For EMNIST, we use the Lorentz model of hyperbolic space $\mathbb { H } ^ { 2 }$ . We convert the point-cloud over $\mathbb { R } ^ { 2 }$ to a distribution over $\mathbb { H } ^ { 2 }$ by simply keeping the x and $y$ coordinates as is and adding a z coordinate such that each point lies on the hyperboloid defined by $- x ^ { 2 } - y ^ { 2 } + z ^ { 2 } = 1$

For the torus experiments, we use KMNIST [ROIS-CODH, 2018], a dataset of $2 8 \times 2 8$ grayscale images of 10 handwritten Kanji characters. As with MNIST and EMNIST, we threshold each pixel and normalize to obtain a point cloud in $\mathbb { R } ^ { 2 }$ . To embed on the 2-torus $\mathbb { T } ^ { 2 } = \mathbb { S } ^ { 1 } \times \mathbb { S } ^ { 1 }$ , we rescale the x and y coordinates from $[ - 1 , 1 ]$ to [0, 2π], treating each as an angular coordinate on the torus.

![](images/88c8ad6ccf880e4bca2edafb611fbb3b596198e1442c143f59280946878f7a29.jpg)

![](images/ab7a1fa6bb713cd42c6df23191d92abe257c458cd67bdc6d2f9ddb28dec6cbf7.jpg)

![](images/89ed07752f755559788d997533b48cfbfc81abb661087d96ac3d80b8b352040a.jpg)  
Fig. S2: Entropic Map Benchmark. a. We evaluate our Riemannian entropic map estimator on constructed examples on the sphere $\mathbb { S } ^ { 2 }$ and hyperbolic space $\mathbb { H } ^ { 2 }$ where the ground-truth map is known. This ground truth arises from shifting an original distribution via the Riemannian gradient of a convex function over each space. We compare the estimation error against the Euclidean entropic map and the unregularized OT map. b. Computational eficiency of our Riemannian entropic map implementation compared the Euclidean map and solving unregularized OT.

D.5. Distributions on a General Mesh (Stanford Bunny). To demonstrate RWEFM on a geometry without closed-form exp/log maps, we generate MNIST digit distributions on the surface of the Stanford bunny. We reduce the mesh resolution to ${ \sim } 7 \mathrm { k }$ faces by vertex clustering, then center it at its centroid and rescale by the largest absolute coordinate to map it isotropically into $[ - 1 , 1 ] ^ { 3 }$ . We equip the mesh with the spectral (biharmonic) premetric of Chen and Lipman [2023]. Concretely, we build the discrete Laplace–Beltrami operator from the cotangent stifness matrix $L -$ with edge weights <sup>1</sup> (cot $\alpha _ { i j } + \cot \beta _ { i j } )$ , where $\alpha _ { i j } , \beta _ { i j }$ are the two angles opposite edge $( i , j )$ and the lumped (barycentric) mass matrix $M = \mathrm { d i a g } ( m _ { i } )$ , where $\begin{array} { r } { m _ { i } = \frac { 1 } { 3 } \sum _ { f \ni i } A _ { f } } \end{array}$ sums one third of the areas $A _ { f }$ of the faces incident to vertex $i .$ We solve the generalized eigenproblem ${ \cal L } \phi _ { i } = \lambda _ { i } \dot { M } \phi _ { i }$ for its $k = 1 0 0$ smallest eigenpairs $( \lambda _ { i } , \phi _ { i } )$ and define the squared distance $\begin{array} { r } { d ^ { 2 } ( x , y ) = \sum _ { i } \lambda _ { i } ^ { - 2 } ( \phi _ { i } ( x ) - \phi _ { i } ( y ) ) ^ { 2 } } \end{array}$ , evaluated at surface points by barycentric interpolation over the nearest triangle. Because the spectral premetric is not geodesic $( \| \nabla d \| \neq 1 )$ , we integrate the premetric conditional vector field of Chen and Lipman [2023] directly for the interpolant, and evaluate the Riemannian entropic map from the distance function and mesh projection only (Appendix A.6). Each MNIST image is binarized with Otsu’s threshold; from the foreground pixels we sample a fixed $N = 1 5 0$ points (with replacement when fewer than 150 are available), normalize them $\mathrm { t o } [ - 1 , 1 ] ^ { 2 }$ , and add small Gaussian jitter. Each resulting planar cloud is then laid onto a fixed tangent chart on the bunny’s flank via the mesh exponential map.

We train one model per digit class {0, 2, 9} for 100,000 steps, using the same network architecture and optimizer as the closed-form experiments. In this experiment, the methods denoted RWEFM and WFM use the sampled OT map rather than the barycentric entropic map used elsewhere in the paper: instead of transporting each source particle to a weighted average of target particles, we assign it to a single target particle drawn from the entropic plan (the same sampled map used for the single-cell experiments). RWEFM and SetRFM operate on the mesh with the spectral metric, whereas WFM and SetFM operate in ambient $\mathbb { R } ^ { 3 }$ and their generated clouds are projected onto the surface for scoring; SetRFM and SetFM use random (identity) couplings without OT. Extended MMD metrics are reported in Table S2.

<table><tr><td rowspan="3">Method</td><td colspan="2">Digit 0</td><td colspan="2">Digit 2</td><td colspan="2">Digit 9</td></tr><tr><td>CD</td><td>EMD</td><td>CD</td><td>EMD</td><td>CD</td><td>EMD</td></tr><tr><td>SetFM</td><td>0.0414</td><td>0.2233</td><td>0.0229</td><td>0.1705</td><td>0.0287</td><td>0.1699</td></tr><tr><td>WFM</td><td>0.0034</td><td>0.0799</td><td>0.0012</td><td>0.0604</td><td>0.0085</td><td>0.1157</td></tr><tr><td>SetRFM</td><td>0.0008</td><td>0.0483</td><td>0.0010</td><td>0.0552</td><td>0.0010</td><td>0.0539</td></tr><tr><td>RWEFM</td><td>0.0007</td><td>0.0468</td><td>0.0010</td><td>0.0538</td><td>0.0009</td><td>0.0527</td></tr></table>

Table S2: MMD on the Stanford bunny mesh (lower is better), with Chamfer (CD) and Earth Mover’s (EMD) ground metrics under the mesh spectral distance, for MNIST digits generated on the bunny. Companion to Table 3; best per column in bold. RWEFM and WFM use the sampled OT map.

D.6. de novo generation of Single-Cell Samples on Spherical spaces. We demonstrate the ability of RWEFM to generate de-novo single-cell samples in the latent space of SCimilarity [Heimberg et al., 2025], which is a foundation model for single-cell data that operates in $\mathbb { S } ^ { 1 2 8 - 1 }$ . SCimilarity was used to embed a single-cell atlas from healthy blood samples of 1, 200 human donors, each consisting of 5, 000 cells profiled with 20, 000 genes across various patient conditions. For benchmarking (as in Tables S4 & 4), we perform unconditional generation, randomly holding out 240 donors and evaluating the quality of 240 generated samples against the true held-out samples. We note that due to the high dimensionality of the space, we used a sampled map instead of the entropic. Briefly, instead of assigning each particle in the source distribution to a weighted average of all particles in the target distribution, we assign each particle to a single particle in the target distribution based on the optimal transport plan. We believe the curse of dimensionality makes the entropic map less efective in this setting, as the entropic map points to unrealistic barycenters of target samples, whereas the sampled map points to actual target samples.

Furthermore, we demonstrate class-conditional generation by training a flow to cell distribution conditioned on tissue status (healthy vs diseased) on an expanded dataset which included pathological samples. In Figure 4, we show that generated samples match the profiles of true samples, and display a marked shift in cellular composition between healthy and diseased samples. This in turn demonstrates the value of using a foundation model, as opposed to Euclidean generation on raw gene expression data or other lower-dimensional Euclidean embeddings. Since RWEFM operates directly in the spherical latent space of SCimilarity, it can leverage the functionality that the foundation model provides, such as accurate cell typing and batch efect robust embeddings.

In Figure S1, we present examples of individual samples generated with RWEFM for diferent tissue and disease combinations. For each generated sample, we match it with the closest sample in the real data by Riemannian Wasserstein distance. We observe that RWEFM can generate realistic tissue samples, showing great potential for downstream biological applications.

D.7. Protein Torsion Angle Generation on Torus. To generate the underlying data, we utilized the MD-CATH dataset [Mirarchi et al., 2024], which contains Molecular Dynamics simulations for domain structures from the CATH database. For each protein in the dataset, we extracted the backbone torsion angles (ϕ, ψ) for every residue at every time step of the simulation. We then aggregated these angle pairs across all residues and time points into a single collection for each protein. Since ϕ and ψ are periodic, this process efectively converts each protein into a single empirical distribution over the flat torus T<sup>2</sup>.

We performed conditional generation by embedding the specific amino acid sequence of each protein using the ESM-2 protein language model to obtain a conditioning vector. We randomly withheld 20% of the proteins as a test set. For these test proteins, we generated their torsion angle distributions (point clouds of size N = 2048) conditioned on their ESM embeddings (Figure 6). Finally, we compared the generated distributions to the ground-truth distributions derived from the MD simulations using standard distributional distance metrics on the torus (Table 4).

D.8. Benchmarking of Riemannian Entropic Map. In this manuscript we introduced the Riemannian analogue of the entropic optimal transport map estimator. In Figure S2, we benchmark the accuracy and computational eficiency of this estimator against the Euclidean entropic map and the unregularized OT map on constructed examples on the sphere S<sup>2</sup> and hyperbolic space H<sup>2</sup>. The ground-truth map is known in these examples, as it arises from shifting an original distribution via the Riemannian gradient of a convex function over each space. In both cases, the source is a (projected) Gaussian distribution centered at the north pole, and the target is obtained by applying the ground-truth map to the source samples. We vary the number of samples and the entropic regularization strength, and report the estimation error of each method in terms of the (spherical/hyperbolic) tangent norm between the estimated and ground-truth map velocities.

Next, we ask how does the quality of the pushed forward distribution vary as a function of map estimator, sample size and regularization strength. In Figure S3, we report the (spherical) EMD between the source samples pushed forward by the estimated map and the target samples, for varying sample sizes and regularization strengths. Here too the source sample is a (projected) Gaussian centered at the north pole, and the target is the gridded sphere as shown in Figure 3. In each experiment, we estimate the map from a limited number of samples, and use the out-of-sample extension of map to push forward the entire $( n = 1 5 , 0 0 0 )$ set of source samples. We see that the Riemannian entropic map consistently outperforms the Euclidean entropic map across sample sizes and regularization strengths.

![](images/3ca04ca5ebf7e70507d26edb119d9c5bfcabe5f39f6d655cccf460782ecf6248.jpg)  
Fig. S3: Entropic map reconstruction error For the spherical grid example from Figure 3, we benchmark the map quality of entropic and euclidean map estimators as a function of sample size and regularization strength. We report the (Spherical) EMD between the source samples pushed forward by the estimated map and the target samples.

D.9. Training Time versus Generation Quality Tradeof. A key practical consideration when applying RWEFM is the tradeof between computational cost and generation quality. To provide users with concrete guidance on this tradeof, we conduct a comprehensive ablation study on the two primary hyperparameters that control both training eficiency and sample quality: the entropic regularization parameter ε and the number of particles n sampled from each distribution during training.

We benchmark RWEFM on the task of generating MNIST digit 3 on the sphere $\mathbb { S } ^ { 2 }$ , systematically varying $\varepsilon \in \{ 0 . 0 0 0 2 , 0 . 0 0 2 , 0 . 0 2 , 0 . 2 \}$ and $n \in \{ 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ across 16 experimental configurations. For each configuration, we train the model for 500,000 steps and measure both total wall-clock training time and generation quality on a held-out test set. Generation quality is assessed using classwise 1-NN deviation (1-NN-D) with both Chamfer Distance (CD) and Earth Mover’s Distance (EMD) as ground metrics, where lower values indicate smaller classwise deviations from 50% accuracy.

The results in Figure S4 reveal a clear and predictable tradeof. As $\varepsilon$ decreases, the entropic optimal transport map becomes less regularized and thus more accurate, requiring more Sinkhorn iterations to converge during training. Similarly, as n increases, the model must process more particles per distribution, necessitating larger kernel sizes in the Sinkhorn algorithm and greater capacity in the transformer feedforward layers. Both factors increase training time—models with $\varepsilon = 0 . 0 0 0 2$ and $n = 2 5 6$ take over 4 hours to train, compared to under an hour for $\varepsilon = 0 . 2$ and $n = 3 2$

However, this computational investment yields substantial improvements in generation quality. The most accurate models use small ε and large $n ,$ while the fastest models exhibit significantly worse quality. Interestingly, the relationship is roughly monotonic: intermediate configurations provide intermediate performance on both axes, allowing practitioners to select hyperparameters that balance their computational budget against their quality requirements. All experiments were conducted on a single NVIDIA B200 GPU, with the reported timings providing practitioners with realistic expectations for their own deployments. This analysis demonstrates that RWEFM ofers a tunable spectrum of performance characteristics.

![](images/6944e8b23c6b410e91325009fb1f4993cf03b9757bd4ceac3aa082532908074f.jpg)  
Fig. S4: Training time versus generation quality tradeof. We benchmark RWEFM on generating MNIST digit 3 on $\mathbb { S } ^ { 2 } .$ , varying the entropic regularization parameter ε and number of particles n per distribution . The quantity plotted on both vertical axes is the classwise 1-NN deviation (1-NN-D), computed using Chamfer Distance and Earth Mover’s Distance and shown against total training time in seconds. Lower scores indicate smaller classwise deviations from 50% accuracy; the minimum is 0. As ε decreases and n increases, training time grows substantially due to increased Sinkhorn iterations required for convergence and larger network capacity needed to process more particles, but generation quality improves markedly as the estimated optimal transport map becomes more accurate. This demonstrates the concrete tradeof between computational cost and sample quality that practitioners can tune based on their application requirements and computational budget.

D.10. Additional Metrics and Error Values. These results complement the main-text benchmarks with additional ground metrics and variability estimates. We first report the overall MMD comparison and the single-cell results, then compare the sampled and entropic RWEFM variants across random seeds. Lower values are better for all metrics; 1-NN-D is reported as the classwise deviation from 0.5.

Benchmark summaries. Table S3 complements the 1-NN-D results in Table 2 with MMD under both CD and EMD ground metrics. The comparison includes point-cloud baselines alongside the flow-matching methods.

Table S3: MMD on MNIST, EMNIST, and KMNIST. Extended metrics for Table 2, using Chamfer Distance (CD) and Earth Mover’s Distance (EMD). Dashes indicate methods not evaluated on $\mathbb { T } ^ { 2 } .$
<table><tr><td></td><td></td><td></td><td>PVD</td><td>PSF</td><td>FM</td><td>SetFM</td><td>WFM</td><td>RFM</td><td>SetRFM</td><td>RWEFM</td></tr><tr><td> $\mathbb { H } ^ { 2 }$ </td><td>h</td><td>CD</td><td> $1 . 6 1 \cdot 1 0 ^ { - 3 }$ </td><td> $3 . 5 0 \cdot 1 0 ^ { - 3 }$ </td><td> $3 . 1 1 \cdot 1 0 ^ { - 2 }$ </td><td> $7 . 7 9 \cdot 1 0 ^ { - 2 }$ </td><td> $2 . 8 7 \cdot 1 0 ^ { - 2 }$ </td><td> $4 . 1 0 \cdot 1 0 ^ { - 2 }$ </td><td> $3 . 1 7 \cdot 1 0 ^ { - 3 }$ </td><td> $3 . 3 9 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td></td><td>EMD CD</td><td> $2 . 8 4 \cdot 1 0 ^ { - 3 }$   $2 . 2 3 \cdot 1 0 ^ { - 3 }$ </td><td> $5 . 2 6 \cdot 1 0 ^ { - 2 }$ </td><td> $6 . 5 9 \cdot 1 0 ^ { - 2 }$ </td><td> $5 . 9 4 \cdot 1 0 ^ { - 2 }$ </td><td> $2 . 6 0 \cdot 1 0 ^ { - 2 }$ </td><td> $5 . 6 8 \cdot 1 0 ^ { - 2 }$ </td><td> $4 . 5 4 \cdot 1 0 ^ { - 3 }$ </td><td> $2 . 8 1 \cdot 1 0 ^ { - 3 }$   $6 . 2 1 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>W</td><td>EMD</td><td> $6 . 2 2 \cdot 1 0 ^ { - 3 }$ </td><td> $7 . 2 2 \cdot 1 0 ^ { - 3 }$   $5 . 6 5 \cdot 1 0 ^ { - 2 }$ </td><td> $6 . 7 7 \cdot 1 0 ^ { - 2 }$   $1 . 2 9 \cdot 1 0 ^ { - 1 }$ </td><td> $8 . 5 8 \cdot 1 0 ^ { - 2 }$   $6 . 4 3 \cdot 1 0 ^ { - 2 }$ </td><td> $6 . 8 7 \cdot 1 0 ^ { - 2 }$   $7 . 5 4 \cdot 1 0 ^ { - 2 }$ </td><td> $2 . 2 0 \cdot 1 0 ^ { - 2 }$   $2 . 6 1 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 0 4 \cdot 1 0 ^ { - 2 }$   $9 . 9 1 \cdot 1 0 ^ { - 3 }$ </td><td> $5 . 4 1 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>y</td><td>CD</td><td> $4 . 0 5 \cdot 1 0 ^ { - 4 }$ </td><td> $3 . 8 8 \cdot 1 0 ^ { - 3 }$ </td><td> $6 . 7 5 \cdot 1 0 ^ { - 2 }$ </td><td> $5 . 4 2 \cdot 1 0 ^ { - 2 }$ </td><td> $4 . 1 6 \cdot 1 0 ^ { - 2 }$ </td><td> $6 . 7 6 \cdot 1 0 ^ { - 2 }$ </td><td> $4 . 9 9 \cdot 1 0 ^ { - 3 }$ </td><td> $2 . 3 3 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td></td><td>EMD</td><td> $1 . 9 6 \cdot 1 0 ^ { - 3 }$ </td><td> $6 . 2 2 \cdot 1 0 ^ { - 2 }$ </td><td> $7 . 5 0 \cdot 1 0 ^ { - 2 }$ </td><td> $4 . 1 2 \cdot 1 0 ^ { - 2 }$ </td><td> $3 . 8 2 \cdot 1 0 ^ { - 2 }$ </td><td> $7 . 1 3 \cdot 1 0 ^ { - 2 }$ </td><td> $6 . 3 7 \cdot 1 0 ^ { - 3 }$ </td><td> $2 . 2 8 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td> $\mathbb { S } ^ { 2 }$ </td><td>3</td><td>CD</td><td> $4 . 0 5 \cdot 1 0 ^ { - 3 }$ </td><td> $5 . 8 5 \cdot 1 0 ^ { - 3 }$ </td><td> $2 . 1 7 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 6 5 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 2 0 \cdot 1 0 ^ { - 2 }$ </td><td> $9 . 2 5 \cdot 1 0 ^ { - 3 }$ </td><td> $2 . 4 7 \cdot 1 0 ^ { - 3 }$ </td><td> $2 . 4 0 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td></td><td>EMD</td><td> $7 . 8 1 \cdot 1 0 ^ { - 3 }$ </td><td> $5 . 1 7 \cdot 1 0 ^ { - 2 }$ </td><td> $2 . 1 5 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 7 3 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 1 6 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 3 1 \cdot 1 0 ^ { - 2 }$ </td><td> $3 . 1 6 \cdot 1 0 ^ { - 3 }$ </td><td> $2 . 2 9 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>4</td><td></td><td> $2 . 3 3 \cdot 1 0 ^ { - 3 }$ </td><td> $4 . 5 3 \cdot 1 0 ^ { - 3 }$ </td><td> $1 . 5 4 \cdot 1 0 ^ { - 2 }$ </td><td> $2 . 8 0 \cdot 1 0 ^ { - 2 }$ </td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>CD</td><td> $4 . 9 0 \cdot 1 0 ^ { - 3 }$ </td><td></td><td>2.54 · 10−2</td><td> $2 . 5 5 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 5 1 \cdot 1 0 ^ { - 2 }$ </td><td> $9 . 2 3 \cdot 1 0 ^ { - 3 }$ </td><td> $2 . 4 9 \cdot 1 0 ^ { - 3 }$ </td><td> $2 . 5 5 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td></td><td>EMD</td><td></td><td> $2 . 2 4 \cdot 1 0 ^ { - 2 }$ </td><td></td><td></td><td> $1 . 3 6 \cdot 1 0 ^ { - 2 }$ </td><td> $2 . 1 3 \cdot 1 0 ^ { - 2 }$ </td><td> $3 . 2 0 \cdot 1 0 ^ { - 3 }$ </td><td> $2 . 3 8 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>8</td><td>CD</td><td> $5 . 9 2 \cdot 1 0 ^ { - 3 }$ </td><td> $6 . 7 1 \cdot 1 0 ^ { - 3 }$ </td><td> $1 . 6 3 \cdot 1 0 ^ { - 2 }$ </td><td> $2 . 4 6 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 8 8 \cdot 1 0 ^ { - 2 }$ </td><td> $8 . 3 5 \cdot 1 0 ^ { - 3 }$ </td><td> $2 . 4 8 \cdot 1 0 ^ { - 3 }$ </td><td> $2 . 5 7 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td></td><td>EMD</td><td> $1 . 0 3 \cdot 1 0 ^ { - 2 }$ </td><td> $5 . 8 0 \cdot 1 0 ^ { - 2 }$ </td><td> $2 . 6 0 \cdot 1 0 ^ { - 2 }$ </td><td> $2 . 1 8 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 6 0 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 5 9 \cdot 1 0 ^ { - 2 }$ </td><td> $3 . 8 1 \cdot 1 0 ^ { - 3 }$ </td><td> $2 . 2 5 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td> $\overline { { \mathbb { T } ^ { 2 } } }$ </td><td>ki</td><td>CD</td><td>一</td><td></td><td> $4 . 5 7 \cdot 1 0 ^ { - 2 }$ </td><td> $3 . 2 9 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 1 8 \cdot 1 0 ^ { - 2 }$ </td><td> $4 . 4 9 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 2 8 \cdot 1 0 ^ { - 2 }$ </td><td> $\overline { { 1 . 4 6 \cdot 1 0 ^ { - 2 } } }$ </td></tr><tr><td></td><td></td><td>EMD</td><td></td><td></td><td> $4 . 5 1 \cdot 1 0 ^ { - 2 }$ </td><td> $3 . 4 8 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 3 9 \cdot 1 0 ^ { - 2 }$ </td><td> $4 . 3 7 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 2 6 \cdot 1 0 ^ { - 2 }$ </td><td> $1 . 4 5 \cdot 1 0 ^ { - 2 }$ </td></tr><tr><td></td><td>na</td><td>CD</td><td>一</td><td>一</td><td> $4 . 6 5 \cdot 1 0 ^ { - 2 }$ </td><td> $5 . 9 1 \cdot 1 0 ^ { - 3 }$ </td><td> $5 . 2 9 \cdot 1 0 ^ { - 3 }$ </td><td> $4 . 5 0 \cdot 1 0 ^ { - 2 }$ </td><td> $7 . 3 0 \cdot 1 0 ^ { - 3 }$ </td><td> $1 . 1 2 \cdot 1 0 ^ { - 2 }$ </td></tr><tr><td></td><td></td><td>EMD</td><td>一</td><td>一</td><td> $5 . 2 4 \cdot 1 0 ^ { - 2 }$ </td><td> $5 . 7 2 \cdot 1 0 ^ { - 3 }$ </td><td> $5 . 2 7 \cdot 1 0 ^ { - 3 }$ </td><td> $5 . 2 8 \cdot 1 0 ^ { - 2 }$ </td><td> $6 . 8 4 \cdot 1 0 ^ { - 3 }$ </td><td> $1 . 0 4 \cdot 1 0 ^ { - 2 }$ </td></tr><tr><td></td><td>ma</td><td>CD</td><td></td><td></td><td> $1 . 7 3 \cdot 1 0 ^ { - 2 }$ </td><td> $6 . 4 7 \cdot 1 0 ^ { - 3 }$ </td><td> $5 . 8 8 \cdot 1 0 ^ { - 3 }$ </td><td> $1 . 6 3 \cdot 1 0 ^ { - 2 }$ </td><td> $3 . 6 8 \cdot 1 0 ^ { - 2 }$ </td><td> $7 . 1 2 \cdot 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td></td><td>EMD</td><td>一 一</td><td>一 一</td><td> $3 . 0 0 \cdot 1 0 ^ { - 2 }$ </td><td> $9 . 4 9 \cdot 1 0 ^ { - 3 }$ </td><td> $6 . 6 5 \cdot 1 0 ^ { - 3 }$ </td><td> $2 . 9 5 \cdot 1 0 ^ { - 2 }$ </td><td> $4 . 1 5 \cdot 1 0 ^ { - 2 }$ </td><td> $7 . 9 2 \cdot 1 0 ^ { - 3 }$ </td></tr></table>

Single-cell sample generation. Table S4 expands the single-cell benchmark in Table 4 to both ground metrics. Reporting 1-NN-D alongside MMD summarizes classwise nearest-neighbor label mixing and kernel-based distributional similarity.

<table><tr><td></td><td colspan="4">Manifold:  $\mathbb { S } ^ { 1 2 8 - 1 }$ </td></tr><tr><td>Method</td><td> $\mathrm { 1 - N N - D \ ( E M D ) }$ </td><td> $\mathrm { 1 - N N - D \Omega \left( C D \right) }$ </td><td> $\mathrm { M M D \ ( E M D ) }$ </td><td> $\mathrm { M M D \ ( C D ) }$ </td></tr><tr><td>SetFM</td><td> $0 . 2 6 7 2 \pm 0 . 0 2 1 8$ </td><td> $0 . 4 4 6 0 \pm 0 . 0 0 7 2$ </td><td> $0 . 1 0 2 6 \pm 0 . 0 4 1 0$ </td><td> $0 . 1 1 7 4 \pm 0 . 0 3 2 3$ </td></tr><tr><td>WFM</td><td> $0 . 3 7 1 4 \pm 0 . 0 9 5 7$ </td><td> $0 . 2 9 4 5 \pm 0 . 0 4 4 2$ </td><td> $0 . 0 3 2 4 \pm 0 . 0 2 2 1$ </td><td> $0 . 0 3 5 5 \pm 0 . 0 1 6 2$ </td></tr><tr><td>SetRFM</td><td> $0 . 4 6 4 5 \pm 0 . 0 2 5 1$ </td><td> $0 . 3 7 5 4 \pm 0 . 0 4 7 0$ </td><td> $0 . 0 3 0 0 \pm 0 . 0 0 6 8$ </td><td> $0 . 0 3 4 1 \pm 0 . 0 1 2 0$ </td></tr><tr><td> $\mathrm { R W E F M }$ </td><td> $0 . 3 0 3 9 \pm 0 . 0 4 5 6$ </td><td> $0 . 1 8 8 9 \pm 0 . 0 5 8 8$ </td><td> $0 . 0 1 4 0 \pm 0 . 0 0 7 4$ </td><td> $0 . 0 1 3 7 \pm 0 . 0 0 7 2$ </td></tr></table>

Table S4: Extended metrics for scRNA-seq generation. Whole-sample generation on $\mathbb { S } ^ { 1 2 8 - 1 }$ , with mean ± standard deviation for classwise 1-NN deviation (1-NN-D) and MMD.  
Variability and transport-map variants. Tables S5 and S6 report mean ± standard deviation over three seeds and five samplings per seed, with separate columns for sampled OT assignments and the barycentric entropic map.

Table S5: 1-NN-D across manifolds. Mean ± standard deviation of the classwise deviation from 0.5; lower is better. Dashes indicate unevaluated configurations.
<table><tr><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">FM</td><td rowspan="2">RFM</td><td rowspan="2">SetFM</td><td rowspan="2">WFM</td><td rowspan="2">PSF</td><td rowspan="2"> $\mathrm { S e t R F M }$ </td><td>RWEFM (sample)</td><td>RWEFM (entropic)</td></tr><tr><td></td><td></td></tr><tr><td rowspan="4"> $\mathbb { H } ^ { 2 }$ </td><td>h</td><td>CD</td><td> $0 . 4 9 2 \pm 0 . 0 0 3$ </td><td> $0 . 4 8 0 \pm 0 . 0 0 3$ </td><td> $0 . 3 7 2 \pm 0 . 0 5 5$ </td><td> $0 . 3 0 2 \pm 0 . 0 4 5$ </td><td> $0 . 1 7 9 \pm 0 . 0 0 2$ </td><td> $0 . 3 0 1 \pm 0 . 0 4 1$ </td><td> $0 . 1 9 2 \pm 0 . 0 1 9$ </td><td> $0 . 1 4 9 \pm 0 . 0 2 2$ </td></tr><tr><td></td><td>EMD</td><td> $0 . 4 8 7 \pm 0 . 0 0 3$ </td><td> $0 . 4 8 3 \pm 0 . 0 0 3$ </td><td> $0 . 3 5 5 \pm 0 . 0 6 0$ </td><td> $0 . 2 7 8 \pm 0 . 0 3 8$ </td><td> $0 . 3 2 5 \pm 0 . 0 0 1$ </td><td> $0 . 3 0 6 \pm 0 . 0 1 3$ </td><td> $0 . 1 5 7 \pm 0 . 0 1 9$ </td><td>0.085 ± 0.012</td></tr><tr><td>W</td><td>CD</td><td> $0 . 4 8 1 \pm 0 . 0 1 5$ </td><td> $0 . 4 3 6 \pm 0 . 0 1 1$ </td><td> $0 . 3 2 1 \pm 0 . 0 2 6$ </td><td> $0 . 4 1 2 \pm 0 . 0 1 4$ </td><td> $0 . 0 5 8 \pm 0 . 0 0 2$ </td><td> $0 . 3 3 7 \pm 0 . 0 0 9$ </td><td> $0 . 2 5 3 \pm 0 . 0 2 3$ </td><td> $0 . 2 2 9 \pm 0 . 0 2 1$ </td></tr><tr><td></td><td>EMD</td><td> $0 . 4 8 2 \pm 0 . 0 1 3$ </td><td> $0 . 4 4 6 \pm 0 . 0 0 7$ </td><td> $0 . 3 3 2 \pm 0 . 0 3 2$ </td><td> $0 . 3 9 3 \pm 0 . 0 2 4$ </td><td> $0 . 2 9 6 \pm 0 . 0 0 1$ </td><td> $0 . 3 2 9 \pm 0 . 0 1 1$ </td><td> $0 . 2 0 1 \pm 0 . 0 2 2$ </td><td> $0 . 1 0 4 \pm 0 . 0 2 3$ </td></tr><tr><td rowspan="4"></td><td>y</td><td>CD</td><td> $0 . 4 7 8 \pm 0 . 0 0 4$ </td><td> $0 . 4 7 2 \pm 0 . 0 0 3$ </td><td> $0 . 3 3 7 \pm 0 . 0 2 6$ </td><td> $0 . 2 9 1 \pm 0 . 0 2 0$ </td><td> $0 . 2 0 6 \pm 0 . 0 0 2$ </td><td> $0 . 3 1 0 \pm 0 . 0 2 6$ </td><td> $0 . 2 2 2 \pm 0 . 0 1 3$ </td><td> $0 . 1 6 2 \pm 0 . 0 1 6$ </td></tr><tr><td></td><td>EMD</td><td> $0 . 4 8 3 \pm 0 . 0 0 4$ </td><td> $0 . 4 8 1 \pm 0 . 0 0 4$ </td><td> $0 . 3 2 6 \pm 0 . 0 1 6$ </td><td> $0 . 2 7 7 \pm 0 . 0 1 6$ </td><td> $0 . 3 0 9 \pm 0 . 0 0 0$ </td><td> $0 . 3 0 8 \pm 0 . 0 1 3$ </td><td> $0 . 2 0 1 \pm 0 . 0 1 2$ </td><td> $0 . 1 1 4 \pm 0 . 0 2 3$ </td></tr><tr><td> $\mathbb { S } ^ { 2 } \mathrm { ~ \textbf ~ { ~ 3 ~ } ~ }$ </td><td>CD</td><td> $0 . 4 0 0 \pm 0 . 0 1 3$ </td><td> $0 . 2 9 5 \pm 0 . 0 1 6$ </td><td> $0 . 2 6 2 \pm 0 . 0 3 0$ </td><td> $0 . 2 9 0 \pm 0 . 0 1 5$ </td><td> $0 . 1 8 8 \pm 0 . 0 0 2$ </td><td> $0 . 3 7 7 \pm 0 . 0 2 3$ </td><td> $0 . 2 5 1 \pm 0 . 0 2 0$ </td><td> $0 . 1 8 2 \pm 0 . 0 1 7$ </td></tr><tr><td>4</td><td>EMD</td><td> $0 . 4 2 2 \pm 0 . 0 0 5$ </td><td> $0 . 3 8 2 \pm 0 . 0 0 7$ </td><td> $0 . 2 6 9 \pm 0 . 0 1 6$ </td><td> $0 . 2 8 5 \pm 0 . 0 1 1$ </td><td> $0 . 3 5 5 \pm 0 . 0 0 0$ </td><td> $0 . 3 7 9 \pm 0 . 0 1 1$ </td><td> $0 . 2 7 0 \pm 0 . 0 1 5$ </td><td> $0 . 1 7 9 \pm 0 . 0 1 5$ </td></tr><tr><td rowspan="4"></td><td></td><td>CD</td><td> $0 . 4 5 2 \pm 0 . 0 1 1$ </td><td> $0 . 4 2 3 \pm 0 . 0 0 8$ </td><td> $0 . 2 9 1 \pm 0 . 0 2 9$ </td><td> $0 . 2 9 1 \pm 0 . 0 3 9$ </td><td> $0 . 1 6 6 \pm 0 . 0 0 1$ </td><td> $0 . 3 7 9 \pm 0 . 0 1 4$ </td><td> $0 . 2 4 5 \pm 0 . 0 1 7$ </td><td> $0 . 1 8 5 \pm 0 . 0 2 0$ </td></tr><tr><td></td><td>EMD</td><td> $0 . 4 6 4 \pm 0 . 0 0 6$ </td><td> $0 . 4 5 3 \pm 0 . 0 0 6$ </td><td> $0 . 2 8 9 \pm 0 . 0 2 8$ </td><td> $0 . 2 7 4 \pm 0 . 0 4 3$ </td><td> $0 . 2 9 0 \pm 0 . 0 0 1$ </td><td> $0 . 3 7 9 \pm 0 . 0 1 0$ </td><td> $0 . 2 6 5 \pm 0 . 0 1 6$ </td><td> $0 . 1 7 7 \pm 0 . 0 1 5$ </td></tr><tr><td></td><td>CD</td><td> $0 . 3 6 5 \pm 0 . 0 2 7$ </td><td> $0 . 3 3 2 \pm 0 . 0 2 7$ </td><td> $0 . 4 0 7 \pm 0 . 1 1 1$ </td><td> $0 . 3 0 8 \pm 0 . 0 7 5$ </td><td> $0 . 2 2 9 \pm 0 . 0 0 2$ </td><td> $0 . 3 9 2 \pm 0 . 0 2 5$ </td><td> $0 . 2 7 8 \pm 0 . 0 2 9$ </td><td> $0 . 1 5 8 \pm 0 . 0 1 1$ </td></tr><tr><td></td><td>EMD</td><td> $0 . 4 0 7 \pm 0 . 0 1 0$ </td><td> $0 . 3 8 3 \pm 0 . 0 0 7$ </td><td> $0 . 3 5 3 \pm 0 . 0 9 3$ </td><td> $0 . 3 6 7 \pm 0 . 0 7 9$ </td><td> $0 . 3 9 0 \pm 0 . 0 0 1$ </td><td> $0 . 3 9 5 \pm 0 . 0 1 4$ </td><td> $0 . 2 9 7 \pm 0 . 0 1 3$ </td><td> $0 . 1 6 3 \pm 0 . 0 2 4$ </td></tr><tr><td rowspan="4">T²</td><td>ki</td><td>CD</td><td> $0 . 4 7 8 \pm 0 . 0 0 2$ </td><td> $0 . 4 7 5 \pm 0 . 0 0 3$ </td><td> $0 . 2 8 1 \pm 0 . 0 2 5$ </td><td> $0 . 2 1 9 \pm 0 . 0 1 4$ </td><td>一</td><td> $0 . 2 6 1 \pm 0 . 0 3 5$ </td><td>一</td><td> $0 . 1 7 1 \pm 0 . 0 1 8$ </td></tr><tr><td></td><td>EMD</td><td> $0 . 4 6 1 \pm 0 . 0 0 3$ </td><td> $0 . 4 5 9 \pm 0 . 0 0 3$ </td><td> $0 . 2 7 9 \pm 0 . 0 1 9$ </td><td> $0 . 1 7 2 \pm 0 . 0 1 3$ </td><td></td><td> $0 . 2 7 9 \pm 0 . 0 2 3$ </td><td></td><td> $0 . 1 7 6 \pm 0 . 0 1 5$ </td></tr><tr><td>na</td><td>CD</td><td> $0 . 4 9 0 \pm 0 . 0 0 1$ </td><td> $0 . 4 9 0 \pm 0 . 0 0 2$ </td><td> $0 . 2 4 0 \pm 0 . 0 1 4$ </td><td> $0 . 1 8 7 \pm 0 . 0 1 2$ </td><td>一</td><td> $0 . 2 3 0 \pm 0 . 0 1 6$ </td><td>一</td><td> $0 . 1 6 1 \pm 0 . 0 1 0$ </td></tr><tr><td></td><td>EMD</td><td> $0 . 4 8 3 \pm 0 . 0 0 1$ </td><td> $0 . 4 8 4 \pm 0 . 0 0 2$ </td><td> $0 . 2 3 5 \pm 0 . 0 1 7$ </td><td> $0 . 1 4 0 \pm 0 . 0 1 2$ </td><td>一</td><td> $0 . 2 3 5 \pm 0 . 0 1 2$ </td><td>一</td><td> $0 . 1 5 5 \pm 0 . 0 1 4$ </td></tr><tr><td rowspan="2"></td><td>ma</td><td>CD</td><td> $0 . 4 6 9 \pm 0 . 0 0 3$ </td><td> $0 . 4 6 8 \pm 0 . 0 0 3$ </td><td> $0 . 2 7 6 \pm 0 . 0 2 0$ </td><td> $0 . 1 7 5 \pm 0 . 0 1 3$ </td><td>一</td><td> $0 . 2 5 6 \pm 0 . 0 3 0$ </td><td>一</td><td> $0 . 1 4 8 \pm 0 . 0 1 4$ </td></tr><tr><td></td><td>EMD</td><td>0.471 ± 0.003</td><td> $0 . 4 7 1 \pm 0 . 0 0 3$ </td><td> $0 . 2 9 4 \pm 0 . 0 1 1$ </td><td> $0 . 1 6 9 \pm 0 . 0 1 3$ </td><td>一</td><td> $0 . 2 8 2 \pm 0 . 0 1 1$ </td><td>一</td><td> $0 . 1 8 1 \pm 0 . 0 2 5$ </td></tr></table>

The MMD results below follow the same dataset and method ordering.

Table S6: MMD across manifolds. Mean ± standard deviation under CD and EMD ground metrics; lower is better. Dashes indicate unevaluated configurations.
<table><tr><td colspan="2"></td><td rowspan="2">FM</td><td rowspan="2">RFM</td><td rowspan="2">SetFM</td><td rowspan="2">WFM</td><td rowspan="2">PSF</td><td rowspan="2">SetRFM</td><td rowspan="2">RWEFM (sample)</td><td rowspan="2">RWEFM (entropic)</td></tr><tr><td>H2 h</td><td>0.0311 ± 0.0104</td></tr><tr><td></td><td></td><td>CD EMD</td><td> $0 . 0 6 5 9 \pm 0 . 0 0 5 3$ </td><td>0.0410 ± 0.0024  $0 . 0 5 6 8 \pm 0 . 0 0 1 4$ </td><td> $0 . 0 7 7 9 \pm 0 . 0 4 2 4$   $0 . 0 5 9 4 \pm 0 . 0 3 2 3$ </td><td> $0 . 0 2 8 7 \pm 0 . 0 1 2 9$   $0 . 0 2 6 0 \pm 0 . 0 0 8 8$ </td><td> $0 . 0 0 3 5 \pm 0 . 0 0 0 0$   $0 . 0 5 2 6 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 0 3 2 \pm 0 . 0 0 1 2$   $0 . 0 0 4 5 \pm 0 . 0 0 0 8$ </td><td> $0 . 0 0 2 7 \pm 0 . 0 0 1 7$   $0 . 0 0 2 5 \pm 0 . 0 0 2 0$ </td><td> $0 . 0 0 3 4 \pm 0 . 0 0 2 4$   $0 . 0 0 2 8 \pm 0 . 0 0 2 3$ </td></tr><tr><td rowspan="4"></td><td>CD</td><td></td><td></td><td> $0 . 0 2 2 0 \pm 0 . 0 0 3 8$ </td><td> $0 . 0 8 5 8 \pm 0 . 0 1 6 0$ </td><td></td><td> $0 . 0 0 7 2 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 1 0 4 \pm 0 . 0 0 5 9$ </td><td> $0 . 0 0 5 1 \pm 0 . 0 0 1 2$ </td><td> $0 . 0 0 6 2 \pm 0 . 0 0 3 0$ </td></tr><tr><td>w</td><td>EMD</td><td> $0 . 0 6 7 7 \pm 0 . 0 3 8 3$ </td><td> $0 . 0 2 6 1 \pm 0 . 0 0 1 5$ </td><td> $0 . 0 6 4 3 \pm 0 . 0 0 7 9$ </td><td> $0 . 0 6 8 7 \pm 0 . 0 0 5 0$ </td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td> $0 . 1 2 8 9 \pm 0 . 1 1 6 7$ </td><td></td><td></td><td> $0 . 0 7 5 4 \pm 0 . 0 0 8 9$ </td><td> $0 . 0 5 6 5 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 0 9 9 \pm 0 . 0 0 3 7$ </td><td> $0 . 0 0 3 9 \pm 0 . 0 0 1 5$ </td><td> $0 . 0 0 5 4 \pm 0 . 0 0 2 0$ </td></tr><tr><td>y</td><td>CD EMD</td><td> $0 . 0 6 7 5 \pm 0 . 0 1 7 8$   $0 . 0 7 5 0 \pm 0 . 0 1 0 2$ </td><td> $0 . 0 6 7 6 \pm 0 . 0 0 6 1$   $0 . 0 7 1 3 \pm 0 . 0 0 2 0$ </td><td> $0 . 0 5 4 2 \pm 0 . 0 1 6 6$ </td><td> $0 . 0 4 1 6 \pm 0 . 0 2 0 7$ </td><td> $0 . 0 0 3 9 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 0 5 0 \pm 0 . 0 0 1 4$ </td><td> $0 . 0 0 2 3 \pm 0 . 0 0 0 6$ </td><td> $0 . 0 0 2 3 \pm 0 . 0 0 1 0$ </td></tr><tr><td>S²</td><td>3</td><td>CD</td><td> $0 . 0 2 1 7 \pm 0 . 0 0 1 0$ </td><td> $0 . 0 0 9 3 \pm 0 . 0 0 0 9$ </td><td> $0 . 0 4 1 2 \pm 0 . 0 0 9 6$ </td><td> $0 . 0 3 8 2 \pm 0 . 0 1 2 6$ </td><td> $0 . 0 6 2 2 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 0 6 4 \pm 0 . 0 0 1 4$ </td><td> $0 . 0 0 2 5 \pm 0 . 0 0 1 6$ </td><td> $0 . 0 0 2 3 \pm 0 . 0 0 1 2$ </td></tr><tr><td rowspan="4"></td><td></td><td></td><td></td><td></td><td> $0 . 0 1 6 5 \pm 0 . 0 0 9 0$ </td><td> $0 . 0 1 2 0 \pm 0 . 0 0 3 4$ </td><td> $0 . 0 0 5 9 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 0 2 5 \pm 0 . 0 0 0 7$ </td><td> $0 . 0 0 3 4 \pm 0 . 0 0 0 5$ </td><td> $0 . 0 0 2 4 \pm 0 . 0 0 0 3$ </td></tr><tr><td>4</td><td>EMD</td><td> $0 . 0 2 1 5 \pm 0 . 0 0 0 6$ </td><td> $0 . 0 1 3 1 \pm 0 . 0 0 0 9$ </td><td> $0 . 0 1 7 3 \pm 0 . 0 0 6 8$ </td><td>0.0116 ± 0.0038</td><td> $0 . 0 5 1 7 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 0 3 2 \pm 0 . 0 0 0 5$ </td><td> $0 . 0 0 3 7 \pm 0 . 0 0 0 9$ </td><td>0.0023 ± 0.0007</td></tr><tr><td>CD</td><td></td><td> $0 . 0 1 5 4 \pm 0 . 0 0 3 5$ </td><td> $0 . 0 0 9 2 \pm 0 . 0 0 1 6$ </td><td>0.0280 ± 0.0111</td><td> $0 . 0 1 5 1 \pm 0 . 0 1 0 9$ </td><td> $0 . 0 0 4 5 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 0 2 5 \pm 0 . 0 0 0 7$ </td><td> $0 . 0 0 3 5 \pm 0 . 0 0 0 5$ </td><td> $0 . 0 0 2 5 \pm 0 . 0 0 0 4$ </td></tr><tr><td></td><td>EMD</td><td>0.0254 ± 0.0019</td><td> $0 . 0 2 1 3 \pm 0 . 0 0 0 3$ </td><td> $0 . 0 2 5 5 \pm 0 . 0 0 9 2$ </td><td>0.0136 ± 0.0093</td><td> $0 . 0 2 2 4 \pm 0 . 0 0 0 0$ </td><td>0.0032 ± 0.0005</td><td>0.0037 ± 0.0009</td><td> $0 . 0 0 2 4 \pm 0 . 0 0 0 7$ </td></tr><tr><td> $\overline { { \mathbb { T } ^ { 2 } \mathrm { ~ \mathbb ~ { k i } ~ } } }$ </td><td>8</td><td>CD EMD</td><td> $0 . 0 1 6 3 \pm 0 . 0 0 3 0$   $0 . 0 2 6 0 \pm 0 . 0 0 1 9$ </td><td> $0 . 0 0 8 4 \pm 0 . 0 0 2 5$   $0 . 0 1 5 9 \pm 0 . 0 0 0 9$ </td><td> $0 . 0 2 4 6 \pm 0 . 0 0 5 7$ </td><td> $0 . 0 1 8 8 \pm 0 . 0 0 4 0$ </td><td> $0 . 0 0 6 7 \pm 0 . 0 0 0 0$ </td><td> $0 . 0 0 2 5 \pm 0 . 0 0 0 6$ </td><td> $0 . 0 0 3 1 \pm 0 . 0 0 0 5$ </td><td> $0 . 0 0 2 6 \pm 0 . 0 0 0 3$ </td></tr><tr><td rowspan="4"></td><td>CD</td><td></td><td></td><td></td><td> $0 . 0 2 1 8 \pm 0 . 0 0 4 8$ </td><td> $0 . 0 1 6 0 \pm 0 . 0 0 6 6$ </td><td> $0 . 0 5 8 0 \pm 0 . 0 0 0 1$ </td><td> $0 . 0 0 3 8 \pm 0 . 0 0 0 6$ </td><td> $0 . 0 0 3 5 \pm 0 . 0 0 0 8$ </td><td> $0 . 0 0 2 2 \pm 0 . 0 0 0 4$ </td></tr><tr><td></td><td>EMD</td><td> $0 . 0 4 5 7 \pm 0 . 0 0 3 9$   $0 . 0 4 5 1 \pm 0 . 0 0 2 1$ </td><td> $0 . 0 4 4 9 \pm 0 . 0 0 3 1$ </td><td> $0 . 0 3 2 9 \pm 0 . 0 1 1 3$ </td><td> $0 . 0 1 1 8 \pm 0 . 0 0 4 9$ </td><td></td><td> $0 . 0 1 2 8 \pm 0 . 0 0 4 0$ </td><td></td><td> $0 . 0 1 4 6 \pm 0 . 0 0 5 8$ </td></tr><tr><td></td><td></td><td></td><td> $0 . 0 4 3 7 \pm 0 . 0 0 2 0$ </td><td> $0 . 0 3 4 8 \pm 0 . 0 1 1 5$ </td><td> $0 . 0 1 3 9 \pm 0 . 0 0 5 8$ </td><td></td><td> $0 . 0 1 2 6 \pm 0 . 0 0 6 2$ </td><td></td><td> $0 . 0 1 4 5 \pm 0 . 0 0 4 9$ </td></tr><tr><td>na</td><td>CD EMD</td><td> $0 . 0 4 6 5 \pm 0 . 0 0 2 0$   $0 . 0 5 2 4 \pm 0 . 0 0 0 9$ </td><td> $0 . 0 4 5 0 \pm 0 . 0 0 1 5$ </td><td> $0 . 0 0 5 9 \pm 0 . 0 0 2 6$ </td><td> $0 . 0 0 5 3 \pm 0 . 0 0 1 0$ </td><td></td><td> $0 . 0 0 7 3 \pm 0 . 0 0 1 5$ </td><td></td><td> $0 . 0 1 1 2 \pm 0 . 0 0 2 7$ </td></tr><tr><td rowspan="4"></td><td></td><td></td><td></td><td> $0 . 0 5 2 8 \pm 0 . 0 0 0 9$ </td><td> $0 . 0 0 5 7 \pm 0 . 0 0 2 2$ </td><td> $0 . 0 0 5 3 \pm 0 . 0 0 0 8$ </td><td></td><td> $0 . 0 0 6 8 \pm 0 . 0 0 1 2$ </td><td></td><td> $0 . 0 1 0 4 \pm 0 . 0 0 2 5$ </td></tr><tr><td>ma CD</td><td></td><td> $0 . 0 1 7 3 \pm 0 . 0 0 1 8$ </td><td> $0 . 0 1 6 3 \pm 0 . 0 0 1 7$ </td><td> $0 . 0 0 6 5 \pm 0 . 0 0 1 5$ </td><td> $0 . 0 0 5 9 \pm 0 . 0 0 0 6$ </td><td></td><td> $0 . 0 3 6 8 \pm 0 . 0 2 3 3$ </td><td></td><td> $0 . 0 0 7 1 \pm 0 . 0 0 0 4$ </td></tr><tr><td>EMD</td><td></td><td> $0 . 0 3 0 0 \pm 0 . 0 0 1 2$ </td><td>0.0295 ± 0.0011</td><td> $0 . 0 0 9 5 \pm 0 . 0 0 1 1$ </td><td> $0 . 0 0 6 7 \pm 0 . 0 0 0 7$ </td><td></td><td> $0 . 0 4 1 5 \pm 0 . 0 2 4 7$ </td><td></td><td> $0 . 0 0 7 9 \pm 0 . 0 0 0 9$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>