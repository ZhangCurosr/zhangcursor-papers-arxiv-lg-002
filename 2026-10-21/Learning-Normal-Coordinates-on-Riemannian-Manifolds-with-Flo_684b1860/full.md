# Learning Normal Coordinates on Riemannian Manifolds with Flow Charts

Edmond Cunningham University of Massachusetts Amherst edmondcunnin@umass.edu

## Abstract

Riemannian manifolds are powerful models for problems with non-Euclidean structure. Recent work in geometric machine learning has focused on training deep neural networks to learn Riemannian metrics from high-dimensional data. However, many geometric algorithms require repeated evaluations of the exponential and logarithmic maps, whose high computational cost restricts the utility of learned Riemannian manifolds in practice. We propose to address this cost by training a conditional invertible neural network, called a flow chart, to approximate normal coordinates around points on a given Riemannian manifold. With this approach, the exponential and logarithmic maps can be approximated by fast Euclidean operations in the model’s learned coordinate space. Our main contribution is an efficient loss function for training flow charts that is nonnegative and, under suitable domain assumptions, is zero precisely when the chart is normal. Across sixteen synthetic and data-derived geometries, flow charts are competitive with a specialized neural geodesic interpolator for logarithmic map estimation, while also being able to estimate the exponential map. When pretrained, flow charts provide a median speedup of 60–70× over numerical baselines at comparable logarithmic map estimation error, and their learned coordinates improve the accuracy of a numerical geodesic solver.

## 1 Introduction

Many real-world problems can be modeled effectively with Riemannian manifolds whose metric encodes feasible state transitions (Diepeveen et al., 2024b; Wurgaft et al., 2026; Sun et al., 2025). These transitions are represented by shortest paths, called geodesics, so that problems like trajectory inference (Kapusniak et al., 2024) and probabilistic inference (Girolami and Calderhead, 2011;´ Williams et al., 2025) can be expressed through geodesic computations. When the geometry is known a priori through domain constraints, geodesics can be computed in closed form or simulated numerically (Lee, 2018; Miolane et al., 2020). However, recent work in geometric machine learning has focused on learning metrics directly from data (Sun et al., 2025; Jones and Lanners, 2026; Bamberger et al., 2026) using neural networks. While these methods offer exciting new avenues for geometric deep learning (Bronstein et al., 2021), these neural metrics present computational challenges that limit their downstream utility.

The two computations that are fundamental to Riemannian algorithms are the exponential and logarithmic maps. These maps play the role of addition and subtraction of points on a manifold, so Riemannian extensions of classical algorithms such as PCA and regression (Fletcher, 2011; Fletcher et al., 2004) evaluate them repeatedly. On general manifolds, neither map has a closed form and both must be solved for numerically. The exponential map is often computed as the solution to a differential equation, like the geodesic equation or Hamilton’s equations (Hairer et al., 2006), both of which can be simulated efficiently with automatic differentiation (Williams et al., 2025). However, the simulation updates depend on the metric’s derivative, which can be unreliable for learned metrics whose value may change rapidly (Arvanitidis et al., 2019). The logarithmic map is the inverse of the exponential map, and requires solving for the geodesic interpolation between two points at a time (Hennig and Hauberg, 2014; Rygaard and Hauberg, 2026). Although this cost can be amortized with neural networks (Kapusniak et al., 2024; Kelshaw and Magri, 2024), these methods´ take both endpoints as input and so they are not able to evaluate the exponential map. We explore a different approach to computing both maps based on learning a coordinate system that simplifies their computations, called Riemann normal coordinates.

![](images/726cbb72c5892e0786071d9002f76bbe907a39c8a9f8ed5eee0c5e6bf24d5437.jpg)  
Figure 1: Illustration of normal coordinate charts and our method, using our notation in Section 3 and Section 4. The purple and green symbols correspond to the same points p and p on the manifold in the top left, and the red curve is the geodesic between them. The tangent-space and coordinate plots depict the same blue region of the manifold in different numerical representations. The top row shows the standard approach to defining normal coordinates through the logarithmic map and the bottom shows our approach. Our method introduces flow charts and trains them to satisfy the identity in purple to learn normal coordinates.

Normal coordinates are an important tool for calculations in Riemannian geometry (Lee, 2018, page 78). Within a suitable neighborhood of a point, geodesics through that point become straight lines in coordinate space, and so the exponential and logarithmic maps can be computed in closed form with addition and subtraction. However, normal coordinates are most often defined in terms of the exponential and logarithmic maps, and methods for computing them rely on the numerical approximations of these maps (Brun et al., 2005; Lin and Zha, 2008). We build on an alternative characterization of normal coordinates that does not depend on either map.

We propose a novel approach for parameterizing and learning normal coordinates on a Riemannian manifold usingflow charts, which are conditional invertible neural networks that represent coordinate charts. Our main contribution is an efficient loss function for training flow charts whose value is nonnegative and equals zero over a geodesic ball if and only if the chart is normal. In our experiments, we first show that flow charts trained with the normal loss can recover the logarithmic map accurately at fixed base points on both synthetic manifolds and geometries derived from data. We then show that charts trained over many base points generalize to unseen ones and achieve median logmap query speedups of 60× over GP Arvanitidis et al. (2019) and 72× over GEORCE Rygaard and Hauberg (2026) across sixteen geometries, and exponential-map query speedups of 1.4–8.2× on the seven high-dimensional geometries at comparable error. Finally, we show that GEORCE can compute more accurate logarithmic maps if applied in the flow chart’s learned coordinate space rather than in data coordinates.

## 2 Related work

Our work is related to methods that learn parametric curves to approximate geodesic interpolations between two points in the setting where the derivatives of the metric are assumed to be unreliable (Arvanitidis et al., 2019). These curve parameterizations include splines (Arvanitidis et al., 2022; Karczewski et al., 2026), the posterior mean of a Gaussian process (Arvanitidis et al., 2019), and neural networks (Kapusniak et al., 2024). While these methods do not learn normal coordinates´ directly, they offer normal coordinate estimates as their initial velocities. This approach is also how classical data-driven methods for learning normal coordinates operate, where solutions are derived from shortest paths on a graph over data points (Brun et al., 2005; Lin and Zha, 2008). These approaches are different from our method in that we learn normal coordinates directly whereas the other methods recover them as a byproduct.

More broadly, flow charts are related to methods that use neural networks to amortize the cost of expensive geometric computations. The most relevant method is metric flow matching (MFM), which parameterizes the geodesic between two endpoints with a neural network (Kapusniak et al., 2024).´ This method learns a geodesic interpolant by minimizing the energy of a parametric curve between the two endpoints under the given metric. Other methods parameterize the geodesic distance function instead. Kelshaw and Magri (2024) and Li et al. (2024) train a network to satisfy the eikonal equation of a given metric. These methods are able to recover the logarithmic map using the gradient of the distance function. Flow charts are also neural network based, but they parameterize normal coordinate charts, which are a more general structure that naturally model the exponential map, logarithmic map and Riemannian distance at once.

Like flow charts, there are other methods that represent a geometry for data using invertible neural network architectures. The pullback metric through a normalizing flow has been used as a geometric model of data (Diepeveen, 2024; Diepeveen et al., 2024a) and is useful for counterfactual explanations (Dombrowski et al., 2021) and interpolations (Fadel et al., 2021). However, each of these method represents its geometry with a single chart. Multi-chart methods instead cover the data with a finite mixture of flow-based charts (Kalatzis et al., 2021; Sidheekh et al., 2022; Paik et al., 2022; Yu et al., 2026), but are limited in the geometries they can represent because they use a fixed number of charts. Our method differs from these approaches in that it is defined on a given Riemannian manifold, instead of defining its own geometry, and parameterizes a continuous family of charts that can be expanded at any point on the manifold, instead of a discrete set of points.

## 3 Background

In this section we review basic notation and terminology that will be used in the paper. We intentionally emphasize the coordinate dependence of geometric objects because our method will parameterize a coordinate system. See Lee (2003, 2018) for a reference on differential geometry.

Let M be an n-dimensional smooth manifold that represents the abstract space that contains data. A coordinate chart is a bijective map from a neighborhood of an abstract point $\mathbf { p } \in \mathcal { M }$ to a subset of $\mathbb { R } ^ { n }$ on which geometric objects admit numeric representations. We assume that the region of M that data is observed in can be represented by one coordinate system, $x \colon \mathcal { M } \to \mathcal { M } _ { x } = \mathbb { R } ^ { n }$ , and $\mathbf { x } = x ( \mathbf { p } )$ denotes the point $\mathbf { p } \in \mathcal { M }$ in the x-coordinate system, so that a dataset contains the x-coordinates of a collection of points. Subscripts denote the coordinate system in which a geometric object on $\mathcal { M }$ is represented. For example, $\dot { V _ { x } } \in \mathbb { R } ^ { n }$ represents the components of a tangent vector $V \in T _ { \mathbf { p } } \mathcal { M }$ , and $\omega _ { x } \in \mathbb { R } ^ { n }$ represents the components of a covector $\omega \in T _ { \mathbf { p } } ^ { * } \mathcal { M }$ . The notation $\begin{array} { r } { D f ( \mathbf { x } ) = \frac { \partial f ( \mathbf { x } ) } { \partial x } } \end{array}$ denotes the Jacobian matrix of f at x.

A Riemannian metric is a point-dependent inner product between two tangent vectors represented by a symmetric positive-definite matrix $g _ { x } ( \mathbf { x } ) \in \mathrm { S P D } ( n )$ at each $\mathbf { x } \in \mathcal { M } _ { x }$ . The cometric defines an inner product between cotangent vectors and is represented by $g _ { x } ( \mathbf { x } ) ^ { - 1 }$ . The pair $( \mathcal { M } , g )$ is called a Riemannian manifold, on which geometric objects like shortest paths, called geodesics, are uniquely defined by g.

A curve $\mathbf { x } _ { ( . ) } \colon \mathbb { R } \to { \mathcal { M } } _ { x }$ starting at an initial point $\mathbf { x } _ { 0 } \in \mathcal { M } _ { x }$ with velocity $V _ { x } \in \mathbb { R } ^ { n }$ is a geodesic if it satisfies an ODE called the geodesic equation. The solution of this ODE is called the exponential map. Given the initial conditions $( \mathbf { x } _ { 0 } , V _ { x } )$ , the exponential map returns the point obtained from simulating the geodesic equation for 1 unit of time, $\mathbf { x } _ { 1 } = \exp _ { \mathbf { x } _ { 0 } } ( V _ { x } )$ . Rescaling the initial velocity with some $t \in \mathbb { R }$ obtains other points on the geodesic, $\mathbf { x } _ { t } : = \exp _ { \mathbf { x } _ { 0 } } ( t V _ { x } )$

Let $B = \{ V _ { x } \in \mathbb { R } ^ { n } \colon V _ { x } ^ { \top } g _ { x } ( \mathbf { x } _ { 0 } ) V _ { x } < \varepsilon ^ { 2 } \} \subset T _ { \mathbf { x } _ { 0 } } \mathcal { M } _ { x }$ be a ball of g-radius ε that is small enough for $\mathrm { e x p } _ { \mathbf { x } _ { \mathrm { ( } } }$ to be a diffeomorphism on B. Then $\begin{array} { r } { \mathcal { X } _ { \mathbf { x } _ { 0 } } : = \exp _ { \mathbf { x } _ { 0 } } ( B ) \subset \mathcal { M } _ { x } } \end{array}$ is a geodesic ball on $\bar { \mathcal { M } } _ { x }$ centered at $\mathbf { x } _ { \mathrm { 0 } }$ . The logarithmic map, $\log _ { \mathbf { x } _ { 0 } } \colon \mathcal { X } _ { \mathbf { x } _ { 0 } }  B$ , is the inverse of $\mathrm { e x p } _ { \mathbf { x } _ { 0 } }$ on $\mathcal { X } _ { \mathbf { x } _ { 0 } }$ . If a point x is outside of the geodesic ball, there may be many geodesics connecting $\mathbf { x } _ { \mathrm { 0 } }$ and x, in which case the logarithmic map is not unique.

Riemann normal coordinates (or “normal coordinates”) are a coordinate system in which geodesics through the base point are radial lines from the origin. Let $\mathcal { X } _ { \mathbf { x } _ { 0 } } \subset \mathcal { M } _ { x }$ be a geodesic ball centered at $\mathbf { x } _ { 0 } \in \mathcal { M } _ { x }$ and let $\mathbf { J } _ { x } \in \mathbb { R } ^ { n \times n }$ be a g-orthogonal matrix at $\mathbf { x } _ { 0 } .$ , which satisfies $\begin{array} { r } { \mathbf { J } _ { x } ^ { \top } g _ { x } ( \mathbf { x } _ { 0 } ) \mathbf { J } _ { x } = I _ { n } } \end{array}$ A normal coordinate chart on ${ \mathcal { X } } _ { \mathbf { x } _ { 0 } }$ maps x to $\mathbf { s } = \mathbf { J } _ { x } ^ { - 1 } V _ { x } \in { S } \subset \mathbb { R } ^ { n }$ , where $V _ { x } = \log _ { \mathbf { x } _ { 0 } } ( \mathbf { x } )$ , so that $\mathbf { x } _ { \mathrm { 0 } }$ maps to 0. See Figure 1. Normal coordinate charts are invertible on their domains, with inverse $\mathbf { s } \mapsto \exp _ { \mathbf { x } _ { 0 } } ( \mathbf { J } _ { x } \mathbf { s } )$ , and are uniquely defined up to the choice of $\mathbf { J } _ { x } .$ Because $D \exp _ { \mathbf { x } _ { 0 } } ( \mathbf { 0 } )$ is the identity map, the Jacobian of the inverse coordinate transformation at the origin is $\mathbf { J } _ { x }$

## 4 Method

We present a novel approach for learning normal coordinate charts on Riemannian manifolds. Our method represents coordinate charts using a conditional invertible neural network and is trained with a loss that renders the flow chart approximately normal.

## 4.1 Flow chart

We parameterize coordinate charts using a conditional invertible neural network, called a flow chart. Definition 1 (Flow chart). Let $\mathcal { M } _ { x } \subset \mathbb { R } ^ { n }$ be the coordinate representation of the manifold M in the x coordinate system. A flow chart is a map $h _ { ( . ) } ^ { \theta } ( . ) \colon { \mathcal { M } } _ { x } \times { \dot { \mathcal { M } } } _ { x } \to \mathbb { R } ^ { n }$ with parameters θ satisfying two conditions.

1. For every base point $\mathbf { x } _ { 0 } \in \mathcal { M } _ { x }$ , the map $h _ { \mathbf { x } _ { 0 } } ^ { \theta } \colon \mathcal { M } _ { x } \to \mathbb { R } ^ { n }$ is a diffeomorphism, with inverse denoted by $f _ { \mathbf { x } _ { 0 } } ^ { \theta } : = ( h _ { \mathbf { x } _ { 0 } } ^ { \theta } ) ^ { - 1 }$

2. Every base point maps to the origin ofits own chart, $h _ { \mathbf { x } } ^ { \theta } ( \mathbf { x } ) = \mathbf { 0 }$ for all $\mathbf { x } \in \mathcal { M } _ { x }$

We denote $\mathbf { z } = h _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { x } )$ as the coordinate representation ofx under the chart $h _ { \mathbf { x } _ { 0 } } ^ { \theta }$

In practice, we use normalizing flow architectures with fast inverses to parameterize flow charts (Rezende and Mohamed, 2015; Papamakarios et al., 2021; Durkan et al., 2019).

Flow charts and normalizing flows are closely related in that they both learn coordinate spaces where expensive computations on $\mathcal { M } _ { x }$ are simplified. From a geometric perspective, normalizing flows learn a coordinate chart so that the pushforward density of a complicated target probability distribution equals the density of a simple prior. This way, sampling from the target reduces to sampling from the prior. Similarly, given a target metric $g _ { x }$ , flow charts will be trained so that the coordinate representation, $g _ { z } ^ { \theta }$ , satisfies an identity that implies the chart is normal. For a flow chart that is centered at $\mathbf { x } _ { 0 } \in \mathcal { M } _ { x } ,$ its coordinate representation of the metric at a point $\mathbf { x } \in \mathcal { M } _ { x }$ is given by the following tensor transformation rule (Lee, 2018):

$$
g _ { z } ^ { \theta } ( \mathbf { z } ) = D f _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { z } ) ^ { \top } g _ { x } ( \mathbf { x } ) D f _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { z } )\tag{1}
$$

Note that the particular coordinate representation depends on the base point $\mathbf { x } _ { \mathrm { 0 } }$ and that we do not write the dependence of $g _ { z } ^ { \theta } ( \mathbf { z } )$ on $\mathbf { x } _ { \mathrm { 0 } }$ for brevity.

## 4.2 Normal flow chart

Our main observation is that normal coordinate charts can be identified as flow charts satisfying the identity in the following proposition.

Proposition 2 (Normal chart identity). Let $( \mathcal { M } , g )$ be a Riemannian manifold with global coordinate system x : $\mathcal { M } \to \mathcal { M } _ { x } , \mathbf { x } _ { 0 } \in \mathcal { M } _ { x }$ be a point, $\mathcal { X } _ { \mathbf { x } _ { 0 } } \subset \mathcal { M } _ { x }$ be a geodesic ball centered at x , ${ { \mathcal U } _ { x } } \subseteq { { \mathcal X } _ { \mathbf { x } _ { 0 } } }$ be an open set containing $\mathbf { x } _ { 0 } ,$ and $h _ { \mathbf { x } _ { 0 } } \colon \mathcal { U } _ { x } \to \mathbb { R } ^ { \tilde { n } }$ be a coordinate chart, where ${ \bf z } = h _ { { \bf x } _ { 0 } } ( { \bf x } )$ represents the coordinate of $\mathbf { x } \in \mathcal { M } _ { x } , h _ { \mathbf { x } _ { 0 } } ( \mathbf { x } _ { 0 } ) = \mathbf { 0 } ,$ , and $\mathcal { Z } _ { \mathbf { x } _ { 0 } } : = h _ { \mathbf { x } _ { 0 } } ( \mathcal { U } _ { x } )$ is star-shaped with respect to the origin. Let $g _ { z } ( \mathbf { z } ) = D f _ { \mathbf { x } _ { 0 } } ( \mathbf { z } ) ^ { \top } g _ { x } ( \mathbf { x } ) D f _ { \mathbf { x } _ { 0 } } ( \mathbf { z } )$ denote the metric ${ \mathit { \ ' } } _ { s { \mathit { z } } }$ components. We say that $h _ { { \bf x } _ { 0 } }$ is normal on U<sub>x</sub> ifit is the restriction to $\mathcal { U } _ { x }$ ofa normal coordinate chart on $\mathcal { X } _ { \mathbf { x } _ { 0 } }$ . Then,

$$
h _ { { \bf x } _ { 0 } } ( \cdot ) i s n o r m a l o n \mathcal { U } _ { x } i f a n d o n l y i f g _ { z } ( { \bf z } ) { \bf z } = { \bf z } , \quad \forall { \bf z } \in \mathcal { Z } _ { { \bf x } _ { 0 } }\tag{2}
$$

This identity is Gauss’s lemma written in normal coordinates and serves as a definition of normal coordinates in physics (Müller et al., 1999), but, to our knowledge, has not been used in a machine learning context. See Section A for a proof.

The advantage of this characterization of normality is that the identity can be checked with any flow chart and only requires matrix-vector products with the metric and Jacobian-vector products with the flow chart. Additionally, as illustrated by Figure 1, when a flow chart satisfies the normal chart identity, the exponential and logarithmic maps can be recovered in closed form on the domain where the identity holds:

$$
\begin{array} { r } { \exp _ { \mathbf { x } _ { 0 } } ( V _ { x } ) = f _ { \mathbf { x } _ { 0 } } ^ { \theta } ( D h _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { x } _ { 0 } ) V _ { x } ) , \qquad \log _ { \mathbf { x } _ { 0 } } ( \mathbf { x } ) = D f _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { 0 } ) h _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { x } ) } \end{array}\tag{3}
$$

We train a flow chart to satisfy Equation (2) at a given $\mathbf { x } _ { \mathrm { 0 } }$ by optimizing its parameters to minimize the residual of the identity.

Proposition 3 (Normal loss). Let $( \mathcal { M } , g )$ be a Riemannian manifold, $h _ { \mathbf { x } _ { 0 } } ^ { \theta }$ be a flow chart with parameters $\theta , \mathcal { U } _ { x } \subset \mathcal { X } _ { \mathbf { x } _ { 0 } }$ be an open set containing $\mathbf { x } _ { 0 } ,$ , and $\rho _ { x } ( \cdot | \mathbf { x } _ { 0 } )$ be a probability density restricted to $\mathcal { U } _ { x }$ , and assume $\rho _ { x } ( \mathbf { x } | \mathbf { x } _ { 0 } ) > 0$ over the set. Additionally, let $\mathcal { U } _ { z } \stackrel { \cdot } { = } h _ { \mathbf { x } _ { 0 } } ( \mathcal { U } _ { x } )$ and let $\rho _ { z } ( \cdot | \mathbf { x } _ { 0 } )$ be the density of the pushforward distribution through $h _ { \mathbf { x } _ { 0 } ; \mathbf { \xi } }$ , and assume $\mathcal { U } _ { z }$ is star-shaped about the origin. The normal loss over ${ \mathcal U } _ { x } f o r h _ { \mathbf x _ { 0 } } ^ { \theta }$ is,

$$
\mathscr { L } _ { n o r m a l } ( \theta ) = \mathbb { E } _ { \mathbf { z } \sim \rho _ { z } } \left[ \mathbf { e } ^ { \top } g _ { z } ^ { \theta } ( \mathbf { z } ) \mathbf { e } + \mathbf { e } ^ { \top } g _ { z } ^ { \theta } ( \mathbf { z } ) ^ { - 1 } \mathbf { e } - 2 \right]\tag{4}
$$

where $\mathbf { e } = \mathbf { z } / \lVert \mathbf { z } \rVert _ { 2 }$ . The integrand is defined to be zero at ${ \bf z = 0 }$ . The normal loss satisfies $\mathcal { L } _ { n o r m a l } \geq 0$ and $\mathcal { L } _ { n o r m a l } = 0$ ifand only $i f h _ { \mathbf { x } _ { 0 } } ^ { \theta }$ is normal on $\mathcal { U } _ { x }$

See Section A for a proof. The integrand is the squared g-norm of the difference between the two sides of the normal chart identity, divided by the squared norm of z to remove the quadratic dependence on the distance from the origin. We discuss the importance of this scaling in Section 4.3. Note that one can train with multiple $\mathbf { x } _ { \mathrm { 0 } }$ if amortization is desired because flow charts are conditional neural networks. The main challenges in training with the normal loss stem from selecting the neighborhood $\mathcal { U } _ { x }$ and distribution $\rho _ { x } ( \cdot \mid \mathbf { x } _ { 0 } )$ . This selection is not trivial because Proposition 3 places geometric conditions on $\mathcal { U } _ { x }$ , and the data might not come with a natural coupling to inform how to sample $\mathbf { x } \sim \rho _ { x } ( \cdot | \mathbf { x } _ { 0 } )$

## 4.3 Practical considerations

Suppose the user has access to a dataset of points $\mathcal { D } = \{ \mathbf { x } ^ { ( i ) } \} _ { i = 1 : N }$ and selects a subset $\mathcal { D } _ { 0 } \subseteq \mathcal { D }$ containing base points $\mathbf { x } _ { 0 } \in \mathcal { D } _ { 0 }$ . The main practical challenge in training with Proposition $^ 3$ is determining how to sample z given $\mathbf { x } _ { 0 } .$ , or equivalently by a change of variables x given $\mathbf { x } _ { \mathrm { 0 } }$ , to estimate the normal loss. To satisfy the assumptions of Proposition 3, this sampler should be restricted to a neighborhood of $\mathbf { x } _ { \mathrm { 0 } }$ on which there exists a normal solution, and have positive density in this region. Let us first consider how to satisfy the latter, as it can be satisfied independently of the former. Given a procedure for sampling pairs of points $( \mathbf { x } _ { 0 } , \mathbf { x } )$ , we propose to draw coordinates to estimate the normal loss by sampling on the chord between $\mathbf { 0 } = h _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { x } _ { 0 } )$ and $\mathbf { z } = h _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { x } )$ , and adding a small amount of Gaussian noise. This amounts to selecting $\rho _ { z }$ to be the distribution of $N ( \cdot \mid t h _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { x } ) , \gamma ^ { 2 } I _ { n } )$ where $t \sim \mathrm { U } ( 0 , 1 )$ ). In our experiments, we select $\gamma$ to be small relative to the chord length (see Section D.1.1). This sampler provides coverage along the endpoint chords and around the origin.

In theory, every x should be restricted to a neighborhood of $\mathbf { x } _ { \mathrm { 0 } }$ that satisfies the assumptions of Proposition 3 so that the minimizer of the normal loss is normal. On simple manifolds, like the sphere or torus, this neighborhood is known a priori, but in general it is unknown and determined by whether or not the exponential is a diffeomorphism on the region. In this general setting, we propose to draw x from D using a heuristic that depends on $\mathbf { x } _ { 0 } .$ like Euclidean $\bar { k }$ nearest neighbors. This choice is motivated by our overarching goal of making flow charts a useful model for estimating solutions to geodesic queries by generalization from the pairs seen at training time. However, decoupling the geometry from the training pairs can cause numerical issues that stem from the relative scale of coordinates seen by the model. Figure 2 illustrates this.

The figure shows two geometries built from data with coordinate curves of learned flow charts, contained within a fixed radius coordinate ball that approximate geodesic balls. Panel (a) shows a diffusion geometry Jones and Lanners (2026), whose metric is designed so that paths through regions supported by data are shorter than paths away from data (Hauberg et al., 2012; Arvanitidis et al., 2016; Kapusniak et al., 2024; Sun et al., 2025). Panel (c) shows the pullback metric through a normalizing´ flow with a uniform prior that is fit to the data (Dombrowski et al., 2021; Diepeveen, 2024), whose Riemannian volume density approximates the probability density of data. The colored background in (c) shows the intensity of the volume density in green, and the inverse volume density in blue for (a). We can see that the size of the coordinate balls and the Riemannian volume density on each geometry relate to the density of data in opposite ways. This behavior has practical implications for the scale of z that the model sees during training for a fixed endpoint sampling strategy. For example, to generate this figure we sample from the Euclidean k nearest neighbors to select x given x<sub>0</sub>. (b) and (d) show the distribution of the norms of the neighbor coordinates for each base point. There is more variation across the relative scales in (b) than (d), meaning that the model and optimization procedure must be robust to this variation. This was our motivation for dividing the normal loss integrand by the squared norm of z, which removes its explicit quadratic dependence on coordinate scale. In early experimentation, we found that it was difficult to scale our approach to geometries over images and suspect that the variation in coordinate scales contributed to these difficulties. We leave the investigation of other endpoint samplers for future work.

![](images/481dd549595a03e2946ea3f8085109b19e6b5760660f31c1f26576aadcb8bb8b.jpg)  
Figure 2: The scale of coordinates seen by a flow chart during amortized training depends jointly on how endpoint pairs are selected and the underlying geometry. (a) shows a diffusion geometry and (c) the pullback metric through a normalizing flow with a uniform prior, overlaid with the coordinate curves of learned flow charts contained in fixed radius coordinate balls. (b) and (d) show the distributions of coordinate norms for the nearest neighbors of each base point. The coordinates of endpoints selected with nearest neighbors have a scale that varies substantially across base points in (b), but is more comparable in (d) because in (a), coordinate balls tend to expand in the data space where observations are concentrated.

## 5 Experiments

Our experiments test the accuracy of charts trained at fixed base points, their performance after pretraining over base points, and their use as coordinates for a numerical solver. The experimental details are given in Section D and the full results for all tested settings are given in Section E.

Geometries We evaluate on six families of manifolds with closed-form exponential and logarithmic maps: spheres, flat tori, hyperbolic spaces, manifolds of symmetric positive-definite matrices, Grassmann manifolds, and special orthogonal groups. Each manifold is used in a low dimensional version, with $n \in \{ 2 , 3 , 4 \}$ , and a high dimensional version, with $n \in \{ 2 8 , 3 0 \}$ , and is observed through a nonlinear change of coordinates so that the observed coordinates are not already normal. We also use three two-dimensional point clouds, a disk, two moons and nested circles, with the RBF metric of Kapusniak et al. (2024), and a diffusion geometry built from alanine dipeptide data´ $( n = 3 0 )$ , learned with Riemannian metric matching (Bamberger et al., 2026). For these four data geometries, the reference logmap is computed with a GP solver (Arvanitidis et al., 2019) in double precision. Each geometry has at most 128 test base points with 1,024 queries per base point on the closed form manifolds and 512 on the data geometries, before discarding queries without a valid reference solution. For each base point $\mathbf { x } _ { 0 } ,$ , let $\mathcal { I } ( \mathbf { x } _ { 0 } )$ denote the indices of the remaining queries and let $\mathcal { T } ( \mathbf { x } _ { 0 } ) \subseteq \mathcal { T } ( \mathbf { x } _ { 0 } )$ ) denote the subset for which a method successfully produces a solution.

Flow chart architecture The flow chart we use in these experiments is a conditional neural spline flow (Durkan et al., 2019) with $L$ rational quadratic spline coupling layers, each with 8 bins on [−10, 10] in both low and high dimensions. Before each coupling layer, we place a parametric elementwise scale and shift that is conditioned on the base point encoder and a dense invertible linear map (with an LU parameterization). Each coupling layer parameterizes its spline using a FiLM conditioned (Perez et al., 2018) residual network with three residual blocks. The conditioning is computed with a separate residual network that encodes the base point into a 64-dimensional feature vector for all coupling layers. The network widths and other architectural choices are given in Section D.1.1.

We optimize the normal loss with the Adam optimizer (Kingma and Ba, 2015) for 2,000,000 steps with a learning rate of $3 \times 1 0 ^ { - 4 }$ with a linear warmup over the first 100 steps and linear decay to 0 for the last half of training, and a batch size of 512. When training over many base points, we found that it was beneficial to include multiple endpoints per base point and used 16 base points with 32 endpoints each. We clip gradients to unit norm and maintain an exponential moving average of the parameters with decay 0.999. We evaluated on held-out validation pairs every 500 steps and retain the checkpoint with the lowest validation loss. The smaller four-layer charts used for the MFM comparison on data geometries used a constant learning rate after warmup and stopped early after 50,000 steps without validation improvement.

## 5.1 Evaluation

Logarithmic map The baselines we consider are all based on geodesic interpolation. The first is metric flow matching (MFM) (Kapusniak et al., 2024), which uses a neural network to parameterize´ a curve that is guaranteed to start at $\mathbf { x } _ { \mathrm { 0 } }$ and end at x. This network uses the same gated residual architecture as the flow chart’s base point encoder. Since flow charts use an expensive, invertible neural network architecture compared to MFM’s unrestricted architecture, we vary the number of blocks and hidden sizes of the MFM to construct two baselines. The first has a similar parameter count of a four-layer flow chart (“parameter matched”), and the second has a comparable compute time (“time matched”). The MFM baselines were trained by minimizing the energy for 2,000,000 steps, using the same optimization procedure as the flow charts. The other two are the Gaussian process solver (GP) of Arvanitidis et al. (2019) and GEORCE (Rygaard and Hauberg, 2026), which solve for the geodesic between the endpoints by refining a curve parameterized by $K \in \{ 4 , 8 , 1 6 , 3 2 \}$ intermediate accelerations and points respectively. MFM serves as a fast, approximate baseline that can benefit from pretraining, while GP and GEORCE are numerical references that must be re-solved for each new pair of endpoints.

We train flow charts using the normal loss, and MFM by minimizing the energy of its curves. In the non-amortized setting, we train a separate chart with $\dot { L } = 3 2$ coupling layers at each of four test base points, using training queries separate from its test queries. For the runtime experiments, we separately vary chart capacity over $L \in \{ 4 , 8 , 1 6 , 3 2 \}$ using the wider coupling networks described in Section D.1.1.

Let $U _ { x } ^ { \left( i \right) }$ be a method’s logmap solution and $V _ { x } ^ { \left( i \right) }$ the reference solution for query i at $\mathbf { x } _ { \mathrm { 0 } }$ . We measure accuracy using the root mean squared error over the successful queries, $\mathcal { E } _ { \mathrm { l o g m a p } } ( \mathbf { x } _ { 0 } )$ , and normalize it by the RMS norm of the reference logmaps to account for differences in scale across geometries to obtain the normalized RMSE, $\mathcal { R } _ { \mathrm { l o g m a p } } ( \mathbf { x } _ { 0 } )$ . Dropping the dependence on $\mathbf { x } _ { \mathrm { 0 } }$ for brevity, these quantities are defined as:

$$
\mathcal { E } _ { \mathrm { l o g m a p } } = \sqrt { \frac { 1 } { | \mathcal { Z } | } \sum _ { i \in \mathcal { I } } \Vert U _ { x } ^ { ( i ) } - V _ { x } ^ { ( i ) } \Vert _ { g } ^ { 2 } } , ~ r _ { \mathrm { r e f } } = \sqrt { \frac { 1 } { | \mathcal { I } | } \sum _ { i \in \mathcal { I } } \Vert V _ { x } ^ { ( i ) } \Vert _ { g } ^ { 2 } } , ~ \mathcal { R } _ { \mathrm { l o g m a p } } = \frac { \mathcal { E } _ { \mathrm { l o g m a p } } } { r _ { \mathrm { r e f } } }
$$

For each geometry, we report the mean of $\mathcal { R } _ { \mathrm { l o g m a p } } ( \mathbf { x } _ { 0 } )$ across base points and its standard error. We also report the wall clock time, $\tau ,$ which is measured in microseconds per query on one NVIDIA A16 in single precision, with compilation and preprocessing excluded.

Exponential map We use the same geometries to evaluate the exponential map. Let $\left( \mathbf { x } _ { 0 } , V _ { x } , \mathbf { x } _ { 1 } \right)$ be a test triplet, where $V _ { x } = \log _ { \mathbf { x } _ { 0 } } ( \mathbf { x } _ { 1 } )$ is known in closed form or estimated by the reference GP solver. Given $\mathbf { x } _ { \mathrm { 0 } }$ and $V _ { x } ,$ , each method is tasked with recovering the endpoint $\mathbf { x } _ { 1 }$ . The baselines simulate the geodesic with initial position $\mathbf { x } _ { \mathrm { 0 } }$ and velocity $V _ { x }$ by integrating Hamilton’s equations with Euler, Heun, Tsit5 (Tsitouras, 2011) and Dopri8 (Prince and Dormand, 1981) with ranges of steps selected to cap the total number of cometric evaluations at 129. We use the implementation of these solvers provided by the Diffrax library (Kidger, 2021).

![](images/78826b95868890e1a3d1702c924eae1e2a85e15e7957b6079edbf1346bac554f.jpg)

Figure 3: Mean logmap normalized RMSE at four fixed base points with error bars indicating one standard error. All methods were evaluated on the same four randomly selected test base points, and use their most favorable settings. The charts use $L = 3 2$ coupling layers, the GP uses $K = 1 6$ intermediate accelerations, and GEORCE uses $K = 3 2$ intermediate knot points. All methods achieve low error, and the individually fitted flow chart achieves error comparable to the GP and GEORCE baselines, with mean error below 0.0032 on every geometry. The pretrained chart has higher error than the individually fitted chart on six of the seven high-dimensional geometries.  
![](images/4bfb2a2c48f2141fd224ed57f25609c55a7cad019dc330a1dad943696015cfa6.jpg)  
Figure 4: Amortized logmap accuracy relative to MFM. Each bar denotes the mean $\mathcal { R } _ { \mathrm { l o g m a p } }$ of the $L = 4$ flow chart or a parameter matched or time matched MFM at the same held out base points. Error bars indicate one standard error. In most settings the flow chart outperforms the baselines, but the parameter matched method performs best in the high dimensional synthetic geometries.

Let $\gamma _ { 1 } ^ { ( i ) }$ be a method’s endpoint solution for query i at $\mathbf { x } _ { \mathrm { 0 } }$ . We measure accuracy using the root mean squared error over the successful queries, $\dot { \mathcal { E } } _ { \mathrm { e x p m a p } } ( \mathbf { x } _ { 0 } )$ , and normalize it by the RMS Euclidean distance from the base point to the reference endpoints to obtain the normalized RMSE, $\mathcal { R } _ { \mathrm { e x p m a p } } ( \mathbf { x } _ { 0 } )$

$$
\mathcal { E } _ { \mathrm { e x p m a p } } = \sqrt { \frac { 1 } { | \mathcal { Z } | } \sum _ { i \in \mathcal { I } } \Vert \gamma _ { 1 } ^ { ( i ) } - \mathbf { x } _ { 1 } ^ { ( i ) } \Vert _ { 2 } ^ { 2 } } , \ r _ { \mathrm { c h o r d } } = \sqrt { \frac { 1 } { | \mathcal { I } | } \sum _ { i \in \mathcal { I } } \Vert \mathbf { x } _ { 1 } ^ { ( i ) } - \mathbf { x } _ { 0 } \Vert _ { 2 } ^ { 2 } } , \ \mathcal { R } _ { \mathrm { e x p m a p } } = \frac { \mathcal { E } _ { \mathrm { e x p m a p } } } { r _ { \mathrm { c h o r d } } }
$$

For this experiment, a query is successful if the method returns a finite endpoint. The reference scale uses all queries in $\mathcal { I } ( \bar { \mathbf { x } } _ { 0 } )$ and is shared across methods. For each geometry, we report the mean of $\mathcal { R } _ { \mathrm { e x p m a p } } ( \mathbf { x } _ { 0 } )$ across base points and its standard error, both multiplied by 100, and the mean wall clock time per query, τ.

![](images/08cfa0b1d804aa79f8ec4d20cfc3b03b5852a28c1fd8b0c7eebe0a3a861a7259.jpg)  
Figure 5: Comparison of time needed to achieve similar error between flow charts and numerical methods. For each comparison, R<sub>⋆</sub> is the lowest error level that both methods can reach among the settings we tested. Each dot is the numerical method’s solve time divided by the chart’s solve time, using each method’s fastest setting with error at most $\mathcal { R } _ { \star }$ . Values above one favor the chart. Flow charts are faster than the numerical baselines, except for exponential-map computation on the low-dimensional data geometries, whose metrics are fast to evaluate.

Runtime comparisons We also compare the runtimes of the methods needed to achieve a common accuracy. For the logarithmic map, we compare pretrained flow charts with GP and GEORCE, and for the exponential map we compare them with the ODE solvers. All methods are timed on the same NVIDIA A16 gpu in the same way, and we sweep over various batch sizes to solve multiple GP and GEORCE queries at once, and report the time for the fastest batch size. For each geometry and comparison, we find the lowest error threshold that both methods can reach among their tested settings, then compare their fastest settings that meet this threshold. See Section D.4 for details.

## 5.2 Results

Figure 3 shows that a flow chart trained at a fixed base point can recover the logarithmic map to a degree comparable to the baselines. Over 4 base points and 1024 and 512 test queries for closed form and data-driven geometries, respectively, the flow chart achieved an average error below 0.0032, and the pretrained flow chart outperformed GEORCE on 11 of the 16 geometries. The chart is more accurate than GEORCE on all sixteen geometries, but slightly less accurate than GP on the high dimensional manifolds. These results empirically validate the correctness of training flow charts for the normal loss.

Figure 4 compares the performance of a flow chart with $L = 4$ against the MFM baselines. The flow chart is more accurate than both MFMs on all nine geometries with $n \leq 4 .$ . On the seven geometries with $n \geq 2 8 .$ , its error is within a factor of 2.1 of time-matched MFM, and is lower on four of them. This could be because the flow chart has a restrictive neural network architecture, so it needs more parameters than MFM’s unconstrained architecture to achieve the same accuracy.

For the logarithmic map, Figure 5 shows that the amortized flow chart is up to 364× faster than GP and 766× faster than GEORCE, with median speedups of 60× and $7 2 \times$ across the sixteen geometries. On the high dimensional closed form manifolds, the speedups are 50–150× against GP and 41–283× against GEORCE. On alanine dipeptide, both comparisons use a threshold of 2.58 on the mean of $1 0 0 \times \mathcal { R } _ { \mathrm { l o g m a p } }$ , which the chart reaches 364× faster than GP and 766× faster than GEORCE.

For the exponential-map comparison in Figure 5, we vary the chart size and the number of steps used by each ODE solver. For each geometry, we select the fastest chart and the fastest setting across all four ODE solvers whose mean $\mathcal { R } _ { \mathrm { e x p m a p } }$ is at most $\mathcal { R } _ { \star } .$ . Each point shows the ODE runtime divided by the chart runtime, so values above one favor the chart. For the exponential map on the seven geometries with $n \geq 2 8$ , Figure 5 shows that flow charts are 1.4–8.2× faster than the ODE solvers to achieve comparable error, and on the two dimensional point clouds, the ODE solvers are faster than the charts. Here, the fastest solver takes 0.14–0.49 times the runtime of the fastest chart. Section D.4.1 reports the exponential-map errors and the comparison at a fixed runtime budget.

## 5.3 Flow chart preconditioner

Finally, we show that pretrained flow charts can be used to refine the logmap solution of GEORCE. Flow charts learn a coordinate space where geodesics are approximated by radial lines through the origin. We hypothesize that this behavior can make the task of solving for geodesics simpler to boost the performance of numerical methods. We test this by comparing the performance of GEORCE in the data space and in the coordinate space. Let $\mathbf { x } _ { \mathrm { 0 } }$ be a test base point and x one of its queries. We encode the query with the pretrained flow chart, and use GEORCE to solve for the geodesic from 0 to $h _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { x } )$ under the coordinate representation of the metric, $g _ { z } ^ { \theta }$ . We use the same test base points and queries, knot counts, stopping tolerance, iteration cap and precision as for GEORCE in the data coordinates. The pretrained flow chart has L = 4 coupling layers and we run GEORCE at knot counts $K \in \{ 4 , 8 , 1 6 , \bar { 3 2 } \}$ . Table 1 shows that solving GEORCE in the learned coordinate space consistently improves its performance.

Table 1: GEORCE in data coordinates and in the coordinates of a pretrained flow chart. Columns vary GEORCE’s knot count K. Each cell reports the mean of $1 0 0 \times \mathcal { R } _ { \mathrm { l o g m a p } }$ across base points in data coordinates followed by an arrow and the value in chart coordinates. The bold numbers above the arrows show the ratio between the two numbers. Solving in chart coordinates consistently lowers error.
<table><tr><td>Geometry</td><td>K = 4</td><td></td><td>K = 8</td><td></td><td>K = 16</td><td></td><td>K = 32</td></tr><tr><td colspan="8">High-dimensional closed-form manifolds</td></tr><tr><td>Sphere 30</td><td>14.44 2.6×</td><td>5.47 7.76</td><td>2.5×</td><td>3.10</td><td>4.14 2.5× 1.66</td><td>2.16</td><td>2.5× 0.87</td></tr><tr><td>Flat torus 30</td><td>3.8× 7.26</td><td>1.91</td><td>4.03 3.7×</td><td>1.10</td><td>2.16 3.6×</td><td>0.60 1.12</td><td>3.6× 0.31</td></tr><tr><td>Hyperbolic 30</td><td>2.2× 11.34</td><td>5.11</td><td>2.2× 6.47 →</td><td>2.92</td><td>2.2× 3.51</td><td>1.57 1.84</td><td>2.3× 0.81 V</td></tr><tr><td>SPD 28</td><td>3.0× 8.28</td><td>2.73</td><td>4.46 2.8×</td><td>1.58</td><td>2.39 2.8×</td><td>0.85 1.25</td><td>2.8× 0.44</td></tr><tr><td>Grassmann 30</td><td>2.9× 8.06</td><td>2.76</td><td>2.8× 4.44 →</td><td>1.59</td><td>2.8× 2.38</td><td>0.86 1.24</td><td>2.8× 0.45</td></tr><tr><td>Rotations 28</td><td>3.4× 7.30</td><td>2.14</td><td>3.3× 4.05</td><td>1.23</td><td>3.2× 2.16</td><td>0.67 1.13</td><td>3.3× 0.35</td></tr><tr><td colspan="8">Data geometries</td></tr><tr><td>Disk</td><td>6.9× 1.66</td><td>0.24 →</td><td>5.2× 1.00</td><td>0.19 V</td><td>3.5× 0.58</td><td>0.16 4</td><td>2.2× 0.34 0.15</td></tr><tr><td>Two moons</td><td>3.64 8.1×</td><td>0.45 &gt;</td><td>6.3× 2.16</td><td>0.34</td><td>1.23 4.1×</td><td>0.30 0.71</td><td>2.5× 0.28</td></tr><tr><td>Nested circles</td><td>8.1× 3.56</td><td>0.44</td><td>6.8× 2.19</td><td>0.32</td><td>4.9× 1.29</td><td>0.26 0.79</td><td>3.3× 0.24</td></tr><tr><td>Alanine dipeptide</td><td>6.0× 8.81</td><td>1.47</td><td>6.0× 4.92</td><td>0.82</td><td>6.0× 2.61</td><td>0.44</td><td>5.9× 1.35 0.23</td></tr></table>

## 6 Conclusion

We introduced flow charts, which are conditional invertible neural networks that learn normal coordi nates on a given Riemannian manifold. Our training objective is non-negative and vanishes precisely at normal coordinate charts, under suitable domain assumptions, enabling direct approximations of both the exponential and logarithmic maps. Across synthetic and data-derived geometries, flow charts approximate the maps on par with numerical methods, while being substantially faster when pretrained on data, and they are comparable in performance to MFM for computing the logarithmic map. The main limitation of our method is that it requires heuristics to select training points and that our method can only answer local geodesic queries. Future work could address these limitations by investigating different training strategies, using flow charts to define their own geometry, and by re-centering charts along longer geodesics to answer queries beyond a single chart’s domain.

## AI use statement

The research ideas and the design of the method and loss functions originated with the authors. We used ChatGPT and Claude to assist with designing and carrying out experiments, writing and debugging code, running jobs, analyzing results, and producing figures. These tools also helped optimize computational performance, including the efficient computation of geodesic accelerations described in the appendix. We used them to suggest draft proofs, check derivations, and proofread the manuscript. We personally checked the experimental setup, results, and proofs, and take responsibility for the final content of this work, including all AI-assisted text, claims, code, and figures.

## References

Georgios Arvanitidis, Lars K Hansen, and Søren Hauberg. A locally adaptive normal distribution. Advances in Neural Information Processing Systems, 29, 2016.

Georgios Arvanitidis, Søren Hauberg, Philipp Hennig, and Michael Schober. Fast and robust shortest paths on manifolds learned from data. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 1506–1515. PMLR, 2019.

Georgios Arvanitidis, Miguel González-Duque, Alison Pouplin, Dimitrios Kalatzis, and Soren Hauberg. Pulling back information geometry. In Gustau Camps-Valls, Francisco J. R. Ruiz, and Isabel Valera, editors, Proceedings ofThe 25th International Conference on Artificial Intelligence and Statistics, volume 151 of Proceedings of Machine Learning Research, pages 4872–4894. PMLR, 28–30 Mar 2022. URL https://proceedings.mlr.press/v151/arvanitidis22b. html.

Jacob Bamberger, Adam Gosztolai, Pierre Vandergheynst, Michael M. Bronstein, and Iolo Jones. Riemannian metric matching for scalable geometric modeling of distributions. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 6108–6127. PMLR, 2026. URL https://proceedings.mlr.press/ v306/bamberger26a.html.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Yash Katariya, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/jax-ml/jax.

Michael M. Bronstein, Joan Bruna, Taco Cohen, and Petar Velickovi ˇ c. Geometric deep learning:´ Grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478, 2021.

Anders Brun, Carl-Fredrik Westin, Magnus Herberthson, and Hans Knutsson. Fast manifold learning based on Riemannian normal coordinates. In Heikki Kalviainen, Jussi Parkkinen, and Arto Kaarna, editors, Image Analysis, pages 920–929, Berlin, Heidelberg, 2005. Springer Berlin Heidelberg. ISBN 978-3-540-31566-7.

Ricardo J. G. B. Campello, Davoud Moulavi, and Jörg Sander. Density-based clustering based on hierarchical density estimates. In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pages 160–172. Springer, 2013.

Willem Diepeveen. Pulling back symmetric Riemannian geometry for data analysis. arXiv preprint arXiv:2403.06612, 2024.

Willem Diepeveen, Georgios Batzolis, Zakhar Shumaylov, and Carola-Bibiane Schönlieb. Scorebased pullback Riemannian geometry: Extracting the data manifold geometry using anisotropic flows. arXiv preprint arXiv:2410.01950, 2024a.

Willem Diepeveen, Carlos Esteve-Yagüe, Jan Lellmann, Ozan Öktem, and Carola-Bibiane Schönlieb. Riemannian geometry for efficient analysis of protein dynamics data. Proceedings of the National Academy ofSciences, 121(33):e2318951121, 2024b.

Manfredo Perdigão do Carmo. Riemannian Geometry. Mathematics: Theory & Applications. Birkhäuser Boston, 1992. ISBN 978-0-8176-3490-2.

Ann-Kathrin Dombrowski, Jan E. Gerken, and Pan Kessel. Diffeomorphic explanations with normalizing flows. In ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models, 2021. URL https://openreview.net/forum?id=ZBR9EpEl6G4.

Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios. Neural spline flows. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/hash/ 7ac71d433f282034e088473244df8c02-Abstract.html.

Bradley Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association, 106(496):1602–1614, 2011.

Samuel G. Fadel, Sebastian Mair, Ricardo da Silva Torres, and Ulf Brefeld. Principled interpolation in normalizing flows. In Machine Learning and Knowledge Discovery in Databases. Research Track, pages 116–131. Springer, 2021.

P. Thomas Fletcher, Conglin Lu, Stephen M. Pizer, and Sarang Joshi. Principal geodesic analysis for the study of nonlinear statistics of shape. IEEE Transactions on Medical Imaging, 23(8):995–1005, 2004.

Thomas Fletcher. Geodesic regression on Riemannian manifolds. In Proceedings of the Third International Workshop on Mathematical Foundations ofComputational Anatomy-Geometrical and Statistical Methods for Modelling Biological Shape Variability, pages 75–86, 2011.

Mark Girolami and Ben Calderhead. Riemann manifold Langevin and Hamiltonian Monte Carlo methods. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73(2): 123–214, 2011. doi: 10.1111/j.1467-9868.2010.00765.x. URL https://rss.onlinelibrary. wiley.com/doi/abs/10.1111/j.1467-9868.2010.00765.x.

Ernst Hairer, Christian Lubich, and Gerhard Wanner. Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations, volume 31 of Springer Series in Computational Mathematics. Springer, second edition, 2006.

Søren Hauberg, Oren Freifeld, and Michael J. Black. A geometric take on metric learning. In Advances in Neural Information Processing Systems, volume 25, 2012.

Philipp Hennig and Søren Hauberg. Probabilistic solutions to differential equations and their application to Riemannian statistics. In Artificial Intelligence and Statistics, pages 347–355. PMLR, 2014.

Iolo Jones and David Lanners. Computing diffusion geometry. arXiv preprint arXiv:2602.06006, 2026.

Dimitris Kalatzis, Johan Ziruo Ye, Jesper Wohlert, and Søren Hauberg. Multi-chart flows. arXiv preprint arXiv:2106.03500, 2021.

Kacper Kapusniak, Peter Potaptchik, Teodora Reu, Leo Zhang, Alexander Tong, Michael Bronstein,´ Avishek J. Bose, and Francesco Di Giovanni. Metric flow matching for smooth interpolations on the data manifold. In Advances in Neural Information Processing Systems, volume 37, pages 135011–135042, 2024.

Rafał Karczewski, Markus Heinonen, Alison Pouplin, Søren Hauberg, and Vikas K. Garg. The spacetime of diffusion models: An information geometry perspective. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=qCsbYJZRA5.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. In Advances in Neural Information Processing Systems, volume 35, 2022.

Daniel Kelshaw and Luca Magri. Computing distances and means on manifolds with a metricconstrained eikonal approach. arXiv preprint arXiv:2404.08754, 2024.

Patrick Kidger. On Neural Differential Equations. PhD thesis, University of Oxford, 2021.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015. URL https://arxiv.org/abs/1412.6980.

John M. Lee. Introduction to Smooth Manifolds, volume 218 of Graduate Texts in Mathematics. Springer, 2003. doi: 10.1007/978-0-387-21752-9.

John M. Lee. Introduction to Riemannian Manifolds, volume 176 of Graduate Texts in Mathematics. Springer, second edition, 2018. ISBN 978-3-319-91754-2. doi: 10.1007/978-3-319-91755-9.

Yiming Li, Jiacheng Qiu, and Sylvain Calinon. A Riemannian take on distance fields and geodesic flows in robotics. arXiv preprint arXiv:2412.05197, 2024.

Tong Lin and Hongbin Zha. Riemannian manifold learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 30(5):796–809, 2008. doi: 10.1109/TPAMI.2007.70735.

Leland McInnes, John Healy, and Steve Astels. hdbscan: Hierarchical density based clustering. Journal ofOpen Source Software, 2(11):205, 2017.

Nina Miolane, Nicolas Guigui, Alice Le Brigant, Johan Mathe, Benjamin Hou, Yann Thanwerdas, Stefan Heyder, Olivier Peltre, Niklas Koep, Hadi Zaatiti, Hatem Hajri, Yann Cabanes, Thomas Gerald, Paul Chauchat, Christian Shewmake, Daniel Brooks, Bernhard Kainz, Claire Donnat, Susan Holmes, and Xavier Pennec. Geomstats: A Python package for Riemannian geometry in machine learning. Journal ofMachine Learning Research, 21(223):1–9, 2020. URL https: //www.jmlr.org/papers/v21/19-027.html.

James Morrill, Patrick Kidger, Lingyi Yang, and Terry Lyons. Neural controlled differential equations for online prediction tasks. arXiv preprint arXiv:2106.11028, 2021. doi: 10.48550/arXiv.2106. 11028. URL https://arxiv.org/abs/2106.11028.

Uwe Müller, Christian Schubert, and Anton van de Ven. A closed formula for the Riemann normal coordinate expansion. General Relativity and Gravitation, 31(11):1759–1768, 1999. doi: 10.1023/ A:1026718301634.

Feliks Nüske, Hao Wu, Jan-Hendrik Prinz, Christoph Wehmeyer, Cecilia Clementi, and Frank Noé. Markov state models from short non-equilibrium simulations — Analysis and correction of estimation bias. The Journal ofChemical Physics, 146(9):094104, 2017.

Taejin Paik, Jaemin Park, and Jung Ho Park. Atlas flow: compatible local structures on the manifold. arXiv preprint arXiv:2210.14149, 2022.

George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22(57):1–64, 2021.

Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Édouard Duchesnay. Scikit-learn: Machine learning in Python. Journal ofMachine Learning Research, 12:2825–2830, 2011.

Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. FiLM: Visual reasoning with a general conditioning layer. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 2018. URL https://ojs.aaai.org/index.php/AAAI/article/view/11671.

Peter J. Prince and John R. Dormand. High order embedded Runge–Kutta formulae. Journal of Computational and Applied Mathematics, 7(1):67–75, 1981.

G. N. Ramachandran, C. Ramakrishnan, and V. Sasisekharan. Stereochemistry of polypeptide chain configurations. Journal ofMolecular Biology, 7(1):95–99, 1963.

Danilo Jimenez Rezende and Shakir Mohamed. Variational inference with normalizing flows. In Francis R. Bach and David M. Blei, editors, Proceedings ofthe 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015, volume 37 of JMLR Workshop and Conference Proceedings, pages 1530–1538. JMLR.org, 2015. URL http://proceedings.mlr. press/v37/rezende15.html.

Frederik Möbius Rygaard and Søren Hauberg. GEORCE: a fast new control algorithm for computing geodesics. Information Geometry, 9(1):167–224, 2026. doi: 10.1007/s41884-026-00192-9.

Sahil Sidheekh, Chris B. Dock, Tushar Jain, Radu Balan, and Maneesh K. Singh. VQ-Flows: Vector quantized local normalizing flows. In Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, volume 180 of Proceedings of Machine Learning Research, pages 1835–1845. PMLR, 2022.

Xingzhi Sun, Danqi Liao, Kincaid MacDonald, Yanlei Zhang, Guillaume Huguet, Guy Wolf, Ian Adelstein, Tim G. J. Rudner, and Smita Krishnaswamy. Geometric-aware generative autoencoders for warped Riemannian metric learning and generative modeling on data manifolds. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 1018–1026. PMLR, 2025.

Charalampos Tsitouras. Runge–Kutta pairs of order 5(4) satisfying only the first column simplifying assumption. Computers & Mathematics with Applications, 62(2):770–775, 2011.

Christoph Wehmeyer and Frank Noé. Time-lagged autoencoders: Deep learning of slow collective variables for molecular kinetics. The Journal ofChemical Physics, 148(24):241703, 2018.

Bernardo Williams, Hanlin Yu, Hoang Phuc Hau Luu, Georgios Arvanitidis, and Arto Klami. Geodesic slice sampler for multimodal distributions with strong curvature. arXiv preprint arXiv:2502.21190, 2025.

Daniel Wurgaft, Can Rager, Matthew Kowal, Vasudev Shyam, Sheridan Feucht, Usha Bhalla, Tal Haklay, Eric Bigelow, Raphael Sarfati, Thomas McGrath, Owen Lewis, Jack Merullo, Noah Goodman, Thomas Fel, Atticus Geiger, and Ekdeep Singh Lubana. Manifold steering reveals the shared geometry of neural network representation and behavior. arXiv preprint arXiv:2605.05115, 2026.

Hanlin Yu, Søren Hauberg, Marcelo Hartmann, Arto Klami, and Georgios Arvanitidis. Learning geometry and topology via multi-chart flows. In Proceedings ofthe 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings ofMachine Learning Research, pages 2962–2970. PMLR, 2026. URL https://proceedings.mlr.press/v300/yu26c.html.

# Supplementary Materials

## A Proofs

Lemma 4 (Contracted Christoffel symbols). Let $( \mathcal { M } , g )$ be a n-dimensional Riemannian manifold, $x \colon \mathcal { M }  \mathcal { M } _ { x }$ be a coordinate system, $\mathbf { x } \in \mathcal { M } _ { x }$ be a point, ${ \cal V } = { \cal V } _ { x } ^ { i } \frac { \partial } { \partial x ^ { i } } \in { \cal T } _ { \bf x } { \cal M } _ { x }$ be a tangent vector and let $\Gamma _ { x } ( \mathbf { x } ) _ { i j } ^ { k }$ denote the Christoffel symbols ofits Levi-Civita connection at $\mathbf { \boldsymbol { x } } \in \mathcal { M } _ { x }$ . Let $\Gamma _ { x } ( \mathbf { x } ) ( V _ { x } , V _ { x } ) \in \mathbb { R } ^ { n }$ denote the contraction $\Gamma _ { x } ( { \bf x } ) ( V _ { x } , V _ { x } ) ^ { k } = \Gamma _ { x } ( { \bf x } ) _ { i j } ^ { k } V ^ { i } V ^ { j }$ . Then

$$
\Gamma _ { x } ( { \bf x } ) ( V _ { x } , V _ { x } ) = g _ { x } ( { \bf x } ) ^ { - 1 } \left[ D _ { V } ( g _ { x } ( { \bf x } ) V _ { x } ) - \frac { 1 } { 2 } \nabla _ { x } \left( V _ { x } ^ { \top } g _ { x } ( { \bf x } ) V _ { x } \right) \right]\tag{5}
$$

where V is constant with respect to the directional derivative $D _ { V }$ and the Euclidean gradient $\nabla _ { x }$

Proof. The Christoffel symbols of the Levi-Civita connection are given by

$$
\Gamma _ { x } ( { \bf x } ) _ { i j } ^ { k } = \frac { 1 } { 2 } g _ { x } ( { \bf x } ) ^ { k m } \left( \frac { \partial g _ { x , j m } ( { \bf x } ) } { \partial x ^ { i } } + \frac { \partial g _ { x , i m } ( { \bf x } ) } { \partial x ^ { j } } - \frac { \partial g _ { x , i j } ( { \bf x } ) } { \partial x ^ { m } } \right)\tag{6}
$$

Contracting with $V _ { x } ^ { i } V _ { x } ^ { j }$ makes the first two terms equal because of the metric is symmetric. Therefore

$$
\Gamma _ { x } ( { \bf x } ) ( V _ { x } , V _ { x } ) ^ { k } = g _ { x } ( { \bf x } ) ^ { k m } \left( \frac { \partial g _ { x , i m } ( { \bf x } ) } { \partial x ^ { j } } V _ { x } ^ { i } V _ { x } ^ { j } - \frac { 1 } { 2 } \frac { \partial g _ { x , i j } ( { \bf x } ) } { \partial x ^ { m } } V _ { x } ^ { i } V _ { x } ^ { j } \right)\tag{7}
$$

Converting this expression to matrix notation, and passing the derivatives through $V _ { x }$ yields the result. □

Proposition 2 (Normal chart identity). Let $( \mathcal { M } , g )$ be a Riemannian manifold with global coordinate system x : $\mathcal { M } \to \mathcal { M } _ { x } , \mathbf { x } _ { 0 } \in \mathcal { M } _ { x }$ be a point, $\mathcal { X } _ { \mathbf { x } _ { 0 } } \subset \mathcal { M } _ { x }$ be a geodesic ball centered at $\mathbf { x } _ { 0 } , \mathcal { U } _ { x } \subseteq \mathcal { X } _ { \mathbf { x } _ { 0 } }$ be an open set containing $\mathbf { x } _ { 0 } ,$ , and $h _ { \mathbf { x } _ { 0 } } \colon \mathcal { U } _ { x } \to \mathbb { R } ^ { \tilde { n } }$ be a coordinate chart, where ${ \bf z } = h _ { { \bf x } _ { 0 } } ( { \bf x } )$ represents the coordinate o $\mathbf { \hat { x } } \in \mathcal { M } _ { x } , h _ { \mathbf { x } _ { 0 } } ( \mathbf { x } _ { 0 } ) = \mathbf { 0 } ,$ , and $\mathcal { Z } _ { \mathbf { x } _ { 0 } } : = h _ { \mathbf { x } _ { 0 } } ( \mathcal { U } _ { x } )$ is star-shaped with respect to the origin. Let $g _ { z } ( \mathbf { z } ) = D f _ { \mathbf { x } _ { 0 } } ( \mathbf { z } ) ^ { \top } g _ { x } ( \mathbf { x } ) D f _ { \mathbf { x } _ { 0 } } ( \mathbf { z } )$ denote the metric ${ \mathit { \ ' } } _ { s { \mathit { z } } }$ components. We say that $h _ { \mathbf { x } _ { 0 } }$ is normal on $\mathcal { U } _ { x } ~ i f i t$ is the restriction to $\mathcal { U } _ { x }$ ofa normal coordinate chart on $\mathcal { X } _ { \mathbf { x } _ { 0 } }$ . Then,

$$
h _ { { \bf x } _ { 0 } } ( \cdot ) i s n o r m a l o n \mathcal { U } _ { x } i f a n d o n l y i f g _ { z } ( { \bf z } ) { \bf z } = { \bf z } , \quad \forall { \bf z } \in \mathcal { Z } _ { { \bf x } _ { 0 } }\tag{2}
$$

Proof. First we will show that when $h _ { \mathbf { x } _ { 0 } }$ is normal, we have that $g _ { z } ( \mathbf { z } ) \mathbf { z } = \mathbf { z }$ using Gauss’s lemma. Since $h _ { \mathbf { x } _ { 0 } }$ is normal, we have that $f _ { \mathbf { x } _ { 0 } } ( \mathbf { z } ) = \exp _ { \mathbf { x } \circ } ( \mathbf { J } _ { x } \mathbf { z } )$ , where $\mathbf { J } _ { x } \overset { \cdot } { = } D f _ { \mathbf { x } _ { 0 } } ( \mathbf { 0 } )$ . By the chain rule, we have that $\begin{array} { r } { D f _ { \mathbf { x } _ { 0 } } ( \mathbf { z } ) = D \exp _ { \mathbf { x } { } ^ { 0 } } ( \mathbf { J } _ { x } \mathbf { z } ) \mathbf { J } _ { { } _ { \boldsymbol { \lambda } } } } \end{array}$ . Suppose $V _ { x } = \mathbf { J } _ { x } \mathbf { z }$ and $W _ { x } = \mathbf { J } _ { \langle }$ w for some w $\mathbf { \Psi } \in \mathbb { R } ^ { n }$ Recall Gauss’s lemma (do Carmo, 1992, Chapter 3, Lemma $3 . 5 )$ , which says that for $\mathbf { x } _ { 0 } \in \mathcal { M } _ { x }$ and $V _ { x } \in \mathbb { R } ^ { n }$ in the domain of $\mathrm { e x p } _ { \mathbf { x } _ { 0 } }$ , we have that for every $W _ { x } \in \mathbb { R } ^ { n }$ , the following expression holds

$$
\begin{array} { r } { \left( D \exp _ { \mathbf { x } _ { 0 } } ( V _ { x } ) V _ { x } \right) ^ { \mathrm { ~ i ~ } } g _ { x } ( \mathbf { x } _ { 1 } ) \left( D \exp _ { \mathbf { x } _ { 0 } } ( V _ { x } ) W _ { x } \right) = V _ { x } ^ { \top } g _ { x } ( \mathbf { x } _ { 0 } ) W _ { x } , \qquad \mathbf { x } _ { 1 } = \exp _ { \mathbf { x } _ { 0 } } ( V _ { x } ) } \end{array}\tag{8}
$$

where $D \exp _ { \mathbf { x } \mathrm { { o } } } ( V _ { x } )$ denotes the Jacobian matrix of the exponential map evaluated at $V _ { x }$ . Then for ${ \bf x } = f _ { { \bf x } _ { 0 } } ( { \bf z } )$ , the LHS can be rewritten as follows

$$
V _ { x } ^ { \top } D \exp _ { \mathbf { x } _ { 0 } } ( V _ { x } ) ^ { \top } g _ { x } ( \mathbf { x } ) D \exp _ { \mathbf { x } _ { 0 } } ( V _ { x } ) W _ { x } = \mathbf { z } ^ { \top } \mathbf { J } _ { x } ^ { \top } D \exp _ { \mathbf { x } _ { 0 } } ( V _ { x } ) ^ { \top } g _ { x } ( \mathbf { x } ) D \exp _ { \mathbf { x } _ { 0 } } ( V _ { x } ) \mathbf { J } _ { x } w\tag{9}
$$

$$
= \mathbf { z } ^ { \top } D f _ { \mathbf { x } _ { 0 } } ( \mathbf { z } ) ^ { \top } g _ { x } ( \mathbf { x } ) D f _ { \mathbf { x } _ { 0 } } ( \mathbf { z } ) w\tag{10}
$$

$$
{ \bf \Pi } = { \bf z } ^ { \top } g _ { z } ( { \bf z } ) w\tag{11}
$$

And the RHS can be written as

$$
V _ { x } ^ { \top } g _ { x } ( \mathbf { x } _ { 0 } ) W _ { x } = \mathbf { z } ^ { \top } \underbrace { \mathbf { J } _ { x } ^ { \top } } g _ { x } ( \mathbf { x } _ { 0 } ) \mathbf { J } _ { x } \mathbf { \Psi } w = \mathbf { z } ^ { \top } w\tag{12}
$$

$$
\mathbf { J } _ { x } { \mathrm { ~ i s ~ } } g { \mathrm { - o r t h o g o n a l } }
$$

Since w is arbitrary, we are left with the identity, $g _ { z } ( \mathbf { z } ) \mathbf { z } = \mathbf { z } .$

Next, we will prove that the identity $g _ { z } ( \mathbf { z } ) \mathbf { z } = \mathbf { z }$ implies that $h _ { \mathbf { x } _ { 0 } }$ is normal. The proof follows in three steps.

⃝1 The identity near 0 forces $g _ { z } ( \mathbf { 0 } ) = I _ { n }$ , and so $\mathbf { J } _ { x } : = D f _ { \mathbf { x } _ { 0 } } ( \mathbf { 0 } )$ is g-orthogonal.

⃝2 The identity on $\mathcal { Z } _ { \mathbf { x } _ { 0 } }$ forces $\Gamma _ { z } ( \mathbf { z } ) ( \mathbf { z } , \mathbf { z } ) = \mathbf { 0 }$ , and so $f _ { \mathbf { x } _ { 0 } } ( \mathbf { z } _ { t } )$ is a geodesic.

$$
\begin{array} { r } { \textcircled { 3 } \textcircled { 1 } + \textcircled { 2 } \implies h _ { \mathbf { x } _ { 0 } } ( \mathbf { x } ) = \mathbf { J } _ { x } ^ { - 1 } \log _ { \mathbf { x } _ { 0 } } ( \mathbf { x } ) } \end{array}
$$

⃝1 First we show that the identity implies that $\mathbf { J } _ { x }$ is g-orthogonal. Let $\mathbf { e } \in \mathbb { R } ^ { n }$ be a unit length vector, $\| \mathbf { e } \| _ { 2 } = 1$ , and let $s > 0$ be small enough for the vector $\mathbf { z } = s \mathbf { e }$ to be contained in $\mathcal { Z } _ { \mathbf { x } _ { 0 } }$ . The identity then reads as $g _ { z } ( s \mathbf { e } ) s \mathbf { e } = s \mathbf { e }$ . Dividing by s, and letting $s  0$ , we get that $g _ { z } ( { \bf 0 } ) { \bf e } = { \bf e }$ . Since e is arbitrary, $g _ { z } ( \mathbf { 0 } ) = I _ { n }$ , and so $D f _ { \mathbf { x } _ { 0 } } ( \mathbf { 0 } ) ^ { \top } g _ { x } ( \mathbf { x } _ { 0 } ) D f _ { \mathbf { x } _ { 0 } } ( \mathbf { 0 } ) = I _ { n }$ , which states that $\mathbf { J } _ { x } : = D f _ { \mathbf { x } _ { 0 } } ( \mathbf { 0 } )$ is g-orthogonal.

⃝2 Next we will show that the identity implies that $f _ { \mathbf { x } _ { 0 } } ( \mathbf { z } _ { t } )$ is a geodesic. Writing the Christoffel contraction from lemma 4 at $V _ { z } = \mathbf { z }$ gives

$$
\Gamma _ { z } ( \mathbf { z } ) ( \mathbf { z } , \mathbf { z } ) = g _ { z } ( \mathbf { z } ) ^ { - 1 } \left[ D _ { \mathbf { z } } \left( g _ { z } ( \mathbf { z } ) \mathbf { z } ^ { \prime } \right) - \frac { 1 } { 2 } \nabla _ { z } \left( { \mathbf { z } ^ { \prime } } ^ { \top } g _ { z } ( \mathbf { z } ) \mathbf { z } ^ { \prime } \right) \right] , \quad \mathrm { w h e r e \ } \mathbf { z } ^ { \prime } = \mathrm { s t o p \_ g r a d i e n t } ( \mathbf { z } )\tag{13}
$$

Under the identity, each of the two terms in the bracket equals zero.

Let $\delta ( { \bf z } ) = g _ { z } ( { \bf z } ) { \bf z } - { \bf z }$ . For the first term, consider the directional derivative $D _ { \mathbf { z } } \delta ( \mathbf { z } )$ . Expanding the expression and rearranging terms yields

$$
D _ { \mathbf { z } } ( g _ { z } ( \mathbf { z } ) \mathbf { z } ^ { \prime } ) = D \delta ( \mathbf { z } ) \mathbf { z } - \delta ( \mathbf { z } )\tag{14}
$$

where $D \delta ( \mathbf { z } )$ is the Jacobian matrix of δ at z.

For the second term, consider $\nabla _ { z } ( \mathbf { z } ^ { \top } \delta ( \mathbf { z } ) )$ . This can be expanded and rearranged to yield

$$
\nabla _ { z } \left( { { \bf { z } ^ { \prime } } ^ { \top } } g _ { z } ( { \bf { z } } ) { \bf { z } ^ { \prime } } \right) = D \delta ( { \bf { z } } ) ^ { \top } { \bf { z } } - \delta ( { \bf { z } } )\tag{15}
$$

Plugging both terms in gives the expression for the Christoffel contraction in terms of δ:

$$
\Gamma _ { z } ( \mathbf { z } ) ( \mathbf { z } , \mathbf { z } ) = g _ { z } ( \mathbf { z } ) ^ { - 1 } \left[ D \delta ( \mathbf { z } ) \mathbf { z } - \frac { 1 } { 2 } D \delta ( \mathbf { z } ) ^ { \top } \mathbf { z } - \frac { 1 } { 2 } \delta ( \mathbf { z } ) \right]\tag{16}
$$

Since $\mathcal { Z } _ { \mathbf { x } _ { 0 } }$ is open, and $\delta ( \mathbf { z } ) = 0$ for all $\mathbf { z } \in \mathcal { Z } _ { \mathbf { x } _ { 0 } } , D \delta ( \mathbf { z } ) = 0$ . So we have $\Gamma _ { z } ( \mathbf { z } ) ( \mathbf { z } , \mathbf { z } ) = \mathbf { 0 }$

Now suppose that $\mathbf { z } \in \mathcal { Z } _ { \mathbf { x } _ { 0 } }$ , and let ${ \bf z } _ { t } = t { \bf z }$ for $t \in ( 0 , 1 ]$ be a curve. Note that $\mathbf { z } _ { t } \in \mathcal { Z } _ { \mathbf { x } _ { 0 } }$ by star shapedness. We will show that $\mathbf { z } _ { t }$ is a geodesic on $( \mathcal { M } _ { z } , g _ { z } )$ . Let $\begin{array} { r } { V _ { t } = \dot { z } _ { t } ^ { i } \frac { \partial } { \partial z ^ { i } } } \end{array}$ be the velocity vector of the curve, and note that $\dot { z } _ { t } = \mathbf { z }$ and $\bar { \dot { z _ { t } } } = 0$ . By definition of a geodesic, z is a geodesic if and only if it has zero covariant acceleration, $\nabla _ { V _ { t } } V _ { t }$ . Expanding the term, we find that its components are 0:

$$
\nabla _ { V _ { t } } V _ { t } = \left( { \dot { z } } _ { t } ^ { i } { \frac { \partial { \dot { z } } _ { t } ^ { k } } { \partial z ^ { i } } } + \Gamma _ { z , i j } ^ { k } { \dot { z } } _ { t } ^ { i } { \dot { z } } _ { t } ^ { j } \right) { \frac { \partial } { \partial z ^ { k } } }\tag{17}
$$

$$
= \left( \underbrace { \frac { d \dot { z } _ { t } ^ { k } } { d t } } _ { \ddot { z } _ { t } = 0 } + \Gamma _ { z } ( \mathbf { z } _ { t } ) ( \dot { z } _ { t } , \dot { z } _ { t } ) ^ { k } \right) \frac { \partial } { \partial z ^ { k } }\tag{18}
$$

$$
= \left( { \frac { 1 } { t ^ { 2 } } } { \underset { = 0 } { \Gamma _ { z } ( \mathbf { z } _ { t } ) ( \mathbf { z } _ { t } , \mathbf { z } _ { t } ) } } ^ { k } \right) { \frac { \partial } { \partial z ^ { k } } }\tag{19}
$$

(20)

The second to last step follows from bilinearity of $\Gamma _ { z } ( { \bf z } ) ( \cdot , \cdot )$ . The covariant acceleration of $\mathbf { z } _ { t }$ is therefore 0 and $\mathbf { z } _ { t }$ is a geodesic. By coordinate invariance, ${ \bf x } _ { t } = f _ { { \bf x } _ { 0 } } ( { \bf z } _ { t } )$ is also a geodesic.

⃝3 Recall the initial value form of geodesics, $\mathbf { x } _ { t } = \exp _ { \mathbf { x } _ { 0 } } ( t V _ { x } )$ , where $V _ { x }$ is an initial velocity and $\mathbf { x } _ { \mathrm { 0 } }$ is the starting point. The initial velocity of our geodesic is $\begin{array} { r } { \frac { d } { d t } | _ { t = 0 } f _ { \mathbf { x } _ { 0 } } ( \mathbf { z } _ { t } ) = D f _ { \mathbf { x } _ { 0 } } ( \mathbf { 0 } ) \mathbf { z } = \mathbf { J } _ { x } \mathbf { z } } \end{array}$ Equating the expressions gives

$$
f _ { \mathbf { x } _ { 0 } } ( \mathbf { z } _ { t } ) = \exp _ { \mathbf { x } _ { 0 } } ( t \mathbf { J } _ { x } \mathbf { z } ) , \quad t \in ( 0 , 1 ]\tag{21}
$$

If we can invert the expression at $t = 1$ , then we are done, but before doing so we must verify that $\mathbf { J } _ { x } \mathbf { z }$ lies in a domain on which $\mathrm { e x p } _ { \mathbf { x } _ { 0 } }$ is invertible.

Recall that the geodesic ball ${ \mathcal { X } } _ { \mathbf { x } _ { 0 } }$ is defined as $\begin{array} { r l r } { \mathcal { X } _ { { \bf x } _ { 0 } } } & { { } = } & { \exp _ { { \bf x } _ { 0 } } ( B ) } \end{array}$ , where $B \ = \ \{ V _ { x } \in $ $\mathbb { R } ^ { n } \colon V _ { x } ^ { \top } g _ { x } ( \mathbf { x } _ { 0 } ) V _ { x } < \varepsilon ^ { 2 } \} \subset T _ { \mathbf { x } _ { 0 } } { \mathcal { M } } _ { x }$ . The proof is done if we show that $\mathbf { J } _ { x } \mathbf { z } \in B$ so that Equation $( 2 1 )$ can be inverted. $\mathbf { B y } \left( \mathbf { 1 } \right) , \mathbf { J } _ { x }$ is g-orthogonal and so $( \mathbf { J } _ { x } \mathbf { z } ) ^ { \top } g _ { x } ( \mathbf { x } _ { 0 } ) ( \mathbf { J } _ { x } \mathbf { z } ) = \mathbf { z } ^ { \top } \mathbf { z }$ , and so an equivalent condition to $\mathbf { J } _ { x } \mathbf { z } \in B$ is $\| \mathbf { z } \| _ { 2 } < \varepsilon$

Consider the curve $X _ { t } = \log _ { \mathbf { x } _ { 0 } } \left( f _ { \mathbf { x } _ { 0 } } ( \mathbf { z } _ { t } ) \right)$ , which is continuous on [0, 1] and takes values in B. Whenever $t \| \mathbf { z } \| _ { 2 } < \varepsilon$ , the vectors $X _ { t }$ and $t \mathbf { J } _ { x } \mathbf { z }$ both lie in B and satisfy $\begin{array} { r } { \mathrm { { { e x p } } } _ { { \bf { x } } _ { 0 } } ( X _ { t } ) = f _ { { \bf { x } } _ { 0 } } ( { \bf { z } } _ { t } ) = } \end{array}$ $\exp _ { { \mathbf { x } } _ { 0 } } ( t { \mathbf { J } } _ { x } { \mathbf { z } } )$ , so the injectivity of $\mathrm { e x p } _ { \mathbf { x } _ { 0 } }$ on B gives $X _ { t } = t \mathbf { J } _ { x } \mathbf { z }$

$\operatorname { I f } \| \mathbf { z } \| _ { 2 } \geq \varepsilon .$ , this would hold for every $t < t ^ { * } = \varepsilon / \| \mathbf { z } \| _ { 2 } \leq 1$ , and continuity would give $X _ { t ^ { * } } = t ^ { * } \mathbf { J } _ { x } \mathbf { z } .$ whose norm satisfies $X _ { t ^ { * } } ^ { \top } g _ { x } ( \mathbf { x } _ { 0 } ) X _ { t ^ { * } } = \varepsilon ^ { 2 }$ and contradicts the strict inequality that defines $B .$ Therefore $\| \mathbf { z } \| _ { 2 } < \varepsilon$ , and so $\mathbf { J } _ { x } \mathbf { z } \in B$ . This means that $f _ { \mathbf { x } _ { 0 } } ( \mathbf { z } ) = \exp _ { \mathbf { x } _ { 0 } } ( \mathbf { J } _ { x } \mathbf { z } ) = \bar { \mathbf { x } }$ admits a unique inverse, giving us the result

$$
h _ { \mathbf { x } _ { 0 } } ( \mathbf { x } ) = \mathbf { J } _ { x } ^ { - 1 } \log _ { \mathbf { x } _ { 0 } } ( \mathbf { x } ) , \quad \mathbf { x } \in \mathcal { U } _ { x }\tag{22}
$$

Proposition 3 (Normal loss). Let $( \mathcal { M } , g )$ be a Riemannian manifold, $h _ { \mathbf { x } _ { 0 } } ^ { \theta }$ be a flow chart with parameters θ, $\mathcal { U } _ { x } ~ \subset ~ \mathcal { X } _ { \mathbf { x } _ { 0 } }$ be an open set containing $\mathbf { x } _ { 0 } ,$ , and $\rho _ { x } ( \cdot | \mathbf { x } _ { 0 } )$ be a probability density restricted to $\mathcal { U } _ { x }$ , and assume $\rho _ { x } ( \mathbf { x } | \mathbf { x } _ { 0 } ) > 0$ over the set. Additionally, let $\mathcal { U } _ { z } \stackrel { \cdot } { = } h _ { \mathbf { x } _ { 0 } } ( \mathcal { U } _ { x } )$ and let $\rho _ { z } ( \cdot | \mathbf { x } _ { 0 } )$ be the density ofthe pushforward distribution through $h _ { \mathbf { x } _ { 0 } }$ , and assume $\mathcal { U } _ { z }$ is star-shaped about the origin. The normal loss over U<sub>x</sub> for $h _ { \mathbf { x } _ { 0 } } ^ { \theta }$ is,

$$
\mathscr { L } _ { n o r m a l } ( \theta ) = \mathbb { E } _ { \mathbf { z } \sim \rho _ { z } } \left[ \mathbf { e } ^ { \top } g _ { z } ^ { \theta } ( \mathbf { z } ) \mathbf { e } + \mathbf { e } ^ { \top } g _ { z } ^ { \theta } ( \mathbf { z } ) ^ { - 1 } \mathbf { e } - 2 \right]\tag{4}
$$

where $\mathbf { e } = \mathbf { z } / \lVert \mathbf { z } \rVert _ { 2 }$ <sub>2</sub>. The integrand is defined to be zero at ${ \bf z = 0 }$ . The normal loss satisfies $\mathcal { L } _ { n o r m a l } \geq 0$ and $\mathcal { L } _ { n o r m a l } = 0 ~ i$ f and only $i f h _ { \mathbf { x } _ { 0 } } ^ { \theta }$ is normal on $\mathcal { U } _ { x }$

Proof. We drop the dependence on θ for brevity. Let $\delta ( { \bf z } ) = g _ { z } ( { \bf z } ) { \bf z } - { \bf z }$ denote the amount by which the normal chart identity Equation (2) fails at $\mathbf { z } \in \mathcal { U } _ { z }$

$\delta ( \mathbf { z } )$ contains the $z$ components of a covector, because $g _ { z } ( \mathbf { z } ) \mathbf { z }$ is the covector $V _ { z } ^ { \flat } \ = \ g _ { z } ( \mathbf { z } ) V _ { z }$ associated with the vector $V _ { z } = \mathbf { z }$ and $\begin{array} { r } { \mathbf { z } = \nabla _ { z } \frac { 1 } { 2 } \| \mathbf { z } \| _ { 2 } ^ { 2 } } \end{array}$ is the gradient of a function, so z contains the components of the covector $d _ { \mathbf { \epsilon } _ { 2 } } ^ { \mathbf { 1 } } \lVert \mathbf { z } \rVert _ { 2 } ^ { 2 }$ . Therefore $\delta ( \mathbf { z } )$ is the difference of two covectors components. This means that its squared g-norm is defined by $\delta ( \mathbf { z } ) ^ { \top } g _ { z } ( \mathbf { z } ) ^ { - 1 } \delta ( \mathbf { z } )$

First we will show that the integrand of Equation (4) is this squared g-norm divided by $\| \mathbf { z } \| _ { 2 } ^ { 2 }$ Expanding the square norm gives

$$
\delta ( \mathbf { z } ) ^ { \top } g _ { z } ( \mathbf { z } ) ^ { - 1 } \delta ( \mathbf { z } ) = \left( g _ { z } ( \mathbf { z } ) \mathbf { z } - \mathbf { z } \right) ^ { \top } g _ { z } ( \mathbf { z } ) ^ { - 1 } \left( g _ { z } ( \mathbf { z } ) \mathbf { z } - \mathbf { z } \right)\tag{23}
$$

$$
= \mathbf { z } ^ { \top } g _ { z } ( \mathbf { z } ) \mathbf { z } - 2 \mathbf { z } ^ { \top } \mathbf { z } + \mathbf { z } ^ { \top } g _ { z } ( \mathbf { z } ) ^ { - 1 } \mathbf { z }\tag{24}
$$

Dividing by $\| \mathbf { z } \| _ { 2 } ^ { 2 }$ and writing $\mathbf { e } = \mathbf { z } / \| \mathbf { z } \| _ { 2 }  { \mathrm { g i v e s } } { \mathbf { e } } ^ { \top } g _ { z } ( \mathbf { z } ) \mathbf { e } + \mathbf { e } ^ { \top } g _ { z } ( \mathbf { z } ) ^ { - 1 } \mathbf { e } - 2$ , which is the integrand of Equation (4) for $\mathbf { z } \neq \mathbf { 0 }$

Since $g _ { z } ( \mathbf { z } ) ^ { - 1 }$ is positive definite, the integrand is a squared norm divided by a positive number, so it is nonnegative for $\mathbf { z } \neq \mathbf { 0 }$ , and it is zero at $\mathbf { z } = \mathbf { 0 }$ by definition. Its expectation under $\rho _ { z }$ is therefore nonnegative, which gives $\mathcal { L } _ { \mathrm { n o r m a l } } \geq 0$ . For $\mathbf { z } \neq \mathbf { 0 }$ , the integrand equals zero if and only if $\delta ( \mathbf { z } ) = \mathbf { 0 }$

Next we will show that ${ \mathcal { L } } _ { \mathrm { n o r m a l } } = 0$ if and only if $h _ { \mathbf { x } _ { 0 } }$ is normal on $\mathcal { U } _ { x }$ . The set $\mathcal { U } _ { x }$ is open, contains $\mathbf { x } _ { 0 } ,$ , lies in the geodesic ball ${ \mathcal { X } } _ { \mathbf { x } _ { 0 } }$ , and has star shaped image $\mathcal { U } _ { z }$ , so by Proposition $2 , h _ { { \bf x } _ { 0 } }$ is normal on $\mathcal { U } _ { x }$ if and only if δ vanishes on $\mathcal { U } _ { z }$ . If δ vanishes on $\mathcal { U } _ { z }$ then the integrand vanishes on the support of $\rho _ { z }$ , so ${ \mathcal { L } } _ { \mathrm { n o r m a l } } = 0$ . Conversely, if ${ \mathcal { L } } _ { \mathrm { n o r m a l } } = 0$ then the integrand, which is nonnegative and continuous on $\mathcal { U } _ { z } \ \backslash \ \{ \bf 0 \}$ , has zero expectation under $\rho _ { z } ,$ and $\rho _ { z }$ is positive on $\mathcal { U } _ { z }$ because $\rho _ { x } ( \cdot | \mathbf { x } _ { 0 } )$ is positive on $\mathcal { U } _ { x }$ and $f _ { \mathbf { x } _ { 0 } }$ is a diffeomorphism. A nonnegative continuous function with zero expectation under a positive density on an open set vanishes on that set, so $\delta ( \mathbf { z } ) = \mathbf { 0 }$ for all $\mathbf { z } \in \mathcal { U } _ { z } \setminus \mathbf { \hat { \{ 0 \} } }$ , and $\delta ( \mathbf { 0 } ) = \mathbf { 0 }$ holds trivially. □

## B Datasets

All of the datasets are split into a training, validation and test set using a (70, 10, 20) split. The size and dimensionality of each dataset are listed below.

<table><tr><td></td><td>Alanine dipeptide</td><td>Disk</td><td>Two moons</td><td>Nested circles</td></tr><tr><td>n</td><td>30</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Elements</td><td>750,000</td><td>500,000</td><td>500,000</td><td>500,000</td></tr></table>

Alanine dipeptide The dataset consists of 750,000 snapshots of a molecule over three 250 ns trajectories (Nüske et al., 2017; Wehmeyer and Noé, 2018). Each snapshot contains the threedimensional positions of 10 heavy atoms (the full molecule has 12 extra hydrogen atoms that are not measured) so that each data point is a vector in $\mathbb { R } ^ { 3 0 }$ . The three trajectories were processed to have the same centroid and orientation, which removes 6 degrees of freedom. We remove the time ordering by shuffling the data, and preprocess by centering and scaling the data by a single scalar so the coordinates have unit root mean square spread.

This dataset is highly structured. As mentioned, the data is preprocessed to shift and orient each trajectory to the same view, so the data manifold is restricted to a 24 dimensional submanifold. Additionally, the mean distance between pairs of atoms across snapshots can be used to determine whether or not each pair is bonded. This yields a graph structure over the atoms in 3d space that characterizes the shape of the molecule. Most of the shape is captured by two main backbone dihedral angles, ϕ and $\psi ,$ which represent most of the variation of the molecule. As such, it is common to project the data points down to the Ramachandran plot (Ramachandran et al., 1963), which is a 2d torus whose axes are each of the dihedral angles. The projection of data onto this plane organizes into clusters that, at a high level, represent the different shapes that the molecule transitions in and out of during a trajectory. For this reason, the Ramachandran plane is often used during evaluation to interpret trajectories.

Synthetic datasets The three synthetic datasets are point clouds of 500,000 points in $\mathbb { R } ^ { 2 }$ , each generated with a fixed random seed. The two moons and the nested circles are generated with the make\_moons, with a noise standard deviation of 0.06, and make\_circles, with an inner radius of 0.5 and a noise standard deviation of 0.04, from Scikit-learn (Pedregosa et al., 2011). We generate the disk data from a density proportional to $\exp ( - 1 . 8 ( x _ { 1 } + 1 ) )$ ) on the unit disk using rejection sampling. After the split, each dataset is centered and scaled by a single scalar so that the training set has unit root mean square spread, and the same transform is applied to the validation and test sets.

Licenses and data permissions The alanine trajectories are publicly provided by the Computational Molecular Biology Group at Freie Universität Berlin through mdshare under CC BY 4.0, which permits reuse with attribution.<sup>1</sup> The remaining datasets are generated synthetically; no human participants are involved. The MFM reference implementation and Jaxtyping use the MIT license; JAX, Equinox, Optax, and Diffrax use Apache 2.0; NumPy, SciPy, and Scikit-learn use BSD 3-Clause; tqdm uses MPL 2.0 and MIT; and Matplotlib uses its PSF-based license.

## C Geometries

We test our method on synthetic metrics that admit closed form solutions, RBF metrics on three point clouds, and a metric learned from alanine dipeptide with Riemannian metric matching (Bamberger et al., 2026).

## C.1 Synthetic metrics

We generate data on the six synthetic manifolds in Table 2, whose exponential maps, logarithmic maps and geodesic distances are known in closed form. Normal coordinates would already give a valid solution at their center, so we apply a nonlinear change of coordinates before presenting the data to our method. For each manifold, we construct one fixed observed coordinate system x, then sample base points $\mathbf { x } _ { \mathrm { 0 } }$ at which we train the normal chart $h _ { \mathbf { x } _ { 0 } } ^ { \theta }$ . The observed coordinates are shared across base points. See Figure 6 for a visual overview.

Let n = dim M and let ι be the injectivity radius. Let $\mathbf { p } \in \mathcal { M }$ for a point and $\mathbf { J } _ { \mathbf { p } } \colon \mathbb { R } ^ { n } \to T _ { \mathbf { p } } { \mathcal { M } }$ for a g-orthonormal frame at that point.

![](images/a616fc8b323e5cbc1069d0f1f7b29c4c3c60f442b517143d4116cfa62f3d7af1.jpg)  
Figure 6: Synthetic data generation. We sample in normal coordinates at the data center $\mathbf { p } _ { \mu }$ (orange), express the same points in normal coordinates at the reference point $\mathbf { p } _ { \mathrm { r e f } }$ (blue), and apply $\Psi$ to obtain observed coordinates (purple). The learned chart is centered at a sampled base point p<sub>0</sub>, whose observed coordinates are $\mathbf { x } _ { 0 } = x ( \mathbf { p } _ { 0 } )$

We center the normal coordinates at a fixed reference point $\mathbf { p } _ { \mathrm { r e f } }$ and sample base points around a data center $\mathbf { p } _ { \mu }$ . We choose $\mathbf { p } _ { \mu }$ at distance ℓ from $\mathbf { p } _ { \mathrm { r e f } }$ in a uniform direction $\mathbf { e } \in S ^ { n - 1 }$ :

$$
\mathbf { p } _ { \mu } = \exp _ { \mathbf { p } _ { \mathrm { r e f } } } ( \ell \mathbf { J } _ { \mathbf { p } _ { \mathrm { r e f } } } \mathbf { e } ) ,
$$

$$
\ell = \left\{ { \begin{array} { l l } { 0 . 3 \iota , } & { \iota < \infty , } \\ { 1 , } & { \iota = \infty } \end{array} } \right.\tag{25}
$$

For a point p in the geodesic ball of radius ι around $\mathbf { p } _ { \mathrm { r e f } } .$ , let s be a normal coordinate chart at p<sub>ref</sub> and x be another coordinate system, defined by

$$
s ( \mathbf { p } ) = \mathbf { J } _ { \mathbf { p } _ { \mathrm { r e f } } } ^ { - 1 } \log _ { \mathbf { p } _ { \mathrm { r e f } } } ( \mathbf { p } ) ,
$$

$$
\mathbf { x } = x ( \mathbf { p } ) = \Psi ( s ( \mathbf { p } ) )\tag{26}
$$

$\Psi \colon \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ is fixed and invertible. When $\iota = \infty .$ , these coordinates cover all of M. We use the metric $g _ { x } = ( x ^ { - 1 } ) ^ { * } g$ in the observed coordinates.

To construct Ψ, we compose four random orthogonal maps $Q _ { 0 } , \ldots , Q _ { 3 }$ and three elementwise nonlinear maps $f _ { 1 } , f _ { 2 } , f _ { 3 }$ in the order

$$
\Psi = Q _ { 3 } \circ f _ { 3 } \circ Q _ { 2 } \circ f _ { 2 } \circ Q _ { 1 } \circ f _ { 1 } \circ Q _ { 0 }\tag{27}
$$

Each $f _ { k }$ applies the following invertible map to an input coordinate c:

$$
c \mapsto m + s \sinh \left( { \mathrm { a r c s i n h } } \left( { \frac { c - m } { s } } \right) + b \right) , \qquad s > 0\tag{28}
$$

We draw $b \sim \mathrm { U n i f o r m } ( - 0 . 8 , 0 . 8 )$ independently for each coordinate and each $f _ { k }$ . m and s are chosen to standardize each coordinate based on a batch of data at initialization. We use the same $\Psi$ for all base points and data splits.

To sample base points, we draw tangent coordinates at the data center $\mathbf { p } _ { \mu }$ from

$$
{ \mathbf u } _ { 0 } \sim N ( { \mathbf 0 } , \sigma ^ { 2 } I _ { n } ) , \quad \mathrm { a n d } \sigma = \frac { 1 } { \sqrt { n } } \left\{ { 0 . 3 \iota } , \ : \ : \ : { \iota < \infty } , \right.\tag{29}
$$

Around each base point we train on a geodesic ball of radius $r _ { \operatorname* { m a x } } = 0 . 2 5 \iota$ when $\iota < \infty , \operatorname { o r } r _ { \operatorname* { m a x } } = 1$ otherwise. For finite ι, we redraw $\mathbf { u } _ { 0 }$ until this entire ball lies inside the domain of x:

$$
d _ { g } \Big ( \mathbf { p } _ { \mathrm { r e f } } , \mathbf { \Pi } \exp _ { \mathbf { p } _ { \mu } } ( \mathbf { J } _ { \mathbf { p } _ { \mu } } \mathbf { u } _ { 0 } ) \Big ) + r _ { \mathrm { m a x } } < \iota\tag{30}
$$

We then generate $n _ { \mathrm { b p } }$ base points from independent accepted draws,

$$
\mathbf { x } _ { 0 } = x \Big ( \exp _ { \mathbf { p } _ { \mu } } ( \mathbf { J } _ { \mathbf { p } _ { \mu } } \mathbf { u } _ { 0 } ) \Big )\tag{31}
$$

At each $\mathbf { x } _ { 0 } ,$ we train $h _ { \mathbf { x } _ { 0 } } ^ { \theta }$ on the geodesic ball

$$
\mathcal { U } _ { x } ( \mathbf { x } _ { 0 } ) = \big \{ \mathbf { x } \in \mathcal { M } _ { x } : d _ { g } ( \mathbf { x } _ { 0 } , \mathbf { x } ) < r _ { \operatorname* { m a x } } \big \}\tag{32}
$$

To sample a point in this ball, we independently draw $\mathbf { v } \sim N ( \mathbf { 0 } , I _ { n } )$ and $t \sim$ Uniform(0, 1), then set

$$
{ \bf u } = r _ { \mathrm { m a x } } t ^ { 1 / n } \frac { { \bf v } } { \| { \bf v } \| _ { 2 } } , \qquad { \bf x } = \exp _ { { \bf x } _ { 0 } } \left( g _ { x } ( { \bf x } _ { 0 } ) ^ { - 1 / 2 } { \bf u } \right) .\tag{33}
$$

This generative process defines $\rho _ { x } ( \cdot \mid \mathbf { x } _ { 0 } )$ with positive density throughout ${ { \mathcal { U } } _ { x } } ( { \bf { x } } _ { 0 } )$ . For each base point, we draw $n _ { \mathrm { t r a i n } } , n _ { \mathrm { v a l } }$ and $n _ { \mathrm { t e s t } }$ training, validation and test points respectively.

## C.2 RBF Metric

The synthetic datasets of the exponential map experiment in Section D.2 use the RBF metric of metric flow matching (MFM) (Kapusniak et al., 2024). This metric is constructed so that geodesics avoid´ regions without data.

$$
g _ { x } ( \mathbf { x } ) = \frac { 1 } { h ( \mathbf { x } ) + \rho } I _ { n } , \qquad h ( \mathbf { x } ) = \sum _ { k = 1 } ^ { K } w _ { k } \exp \left( - \frac { \lambda _ { k } } { 2 } \| \mathbf { x } - \mathbf { c } _ { k } \| _ { 2 } ^ { 2 } \right)\tag{34}
$$

where h is a radial basis function network with $K = 5 0$ centers $\mathbf { c } _ { k }$ , bandwidths $\lambda _ { k }$ and weights $w _ { k }$ and $\rho > 0$ is an offset. Although Kapusniak et al. (2024) state a more general diagonal metric, their´ released code fits a single network and applies it to every coordinate.

The centers are selected as the $K$ -means centroids of the training set. Let $\sigma _ { k }$ denote the root mean square spread of the k-th cluster per coordinate. We set $\lambda _ { k } = 1 / ( 2 ( \kappa \sigma _ { k } ) ^ { 2 } )$ with $\kappa = 1 . 5$

Let $\mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( M ) }$ denote $M = 5 0 { , } 0 0 0$ points drawn at random from the training set. We select the weights by solving the following optimization problem

$$
\operatorname* { m i n } _ { w _ { 1 } , \ldots , w _ { K } \geq 1 0 ^ { - 4 } } J , \qquad J = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \big ( 1 - h ( \mathbf { x } ^ { ( i ) } ) \big ) ^ { 2 } .\tag{35}
$$

The fit makes $h \approx 1$ on the data, so $g _ { x }$ is close to $( 1 + \rho ) ^ { - 1 } I _ { n }$ there and grows to $\rho ^ { - 1 } I _ { n }$ away from it. This makes paths that leave the data have large lengths, so geodesics will tend to stay close to the data. We set $\rho = ( 1 - J ) / 2 .$ 75 following MFM, where J is evaluated at the fitted weights and averaged over the validation set in place of the M training points.

## C.3 Diffusion geometry

Figure 2(a) was built with non-parametric diffusion geometries built from 2d point clouds (Jones and Lanners, 2026, Section 3.1). The pointclouds we use come from $m = 5 0 , 0 0 0$ samples from three toy datasets depicting an uneven concentration on a disk, two moons, and nested circles Section B. The disk has density proportional $\mathbf { \sigma } _ { \cdot 0 } \exp ( - 1 . 8 x ^ { 1 } ) \mathbf { o n } \| \mathbf { x } \| _ { 2 } < 1$ and both the two moons and nested circles have Gaussian noise standard deviations 0.06 and 0.04, respectively, with circle radii 1 and 0.5.

Let $\{ \mathbf { x } ^ { ( j ) } \} _ { i = 1 } ^ { m } \subset \mathbb { R } ^ { n }$ denote the point cloud. We build a Markov chain over the cloud from a variable-bandwidth Gaussian kernel supported on each point’s 32 nearest neighbors,

$$
k ( \mathbf { x } , \mathbf { y } ) = \exp { \left( - \frac { \| \mathbf { x } - \mathbf { y } \| _ { 2 } ^ { 2 } } { \epsilon \rho ( \mathbf { x } ) \rho ( \mathbf { y } ) } \right) } , \qquad \tilde { \mathbf { K } } _ { j l } = \frac { k ( \mathbf { x } ^ { ( j ) } , \mathbf { x } ^ { ( l ) } ) } { \left( q ( \mathbf { x } ^ { ( j ) } ) q ( \mathbf { x } ^ { ( l ) } ) \right) ^ { ( 2 - \partial ) / { 4 } } } , \qquad \mathbf { P } _ { j l } = \frac { \tilde { \mathbf { K } } _ { j l } } { \sum _ { l ^ { \prime } = 1 } ^ { m } \tilde { \mathbf { K } } _ { j l ^ { \prime } } } ,\tag{36}
$$

Table 2: Overview of synthetic manifolds. $V , W$ are tangent vectors on each manifold. Dims lists the tested intrinsic dimensions.
<table><tr><td>Space / element</td><td>Metric</td><td>Exp. map</td><td>Log. map</td><td>Distance</td><td>Inj. radius</td><td>Dims</td></tr><tr><td>Sphere</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathbf { p } _ { 0 } , \mathbf { p } _ { 1 } \in S ^ { n }$ </td><td> $V ^ { \top } W$ </td><td> $\begin{array} { r } { \cos ( r ) \mathbf { p } _ { 0 } + \frac { \sin r } { r } V } \end{array}$ </td><td> $\begin{array} { r } { \frac { \theta } { \sin \theta } ( \mathbf { p } _ { 1 } - \cos ( \theta ) \mathbf { p } _ { 0 } ) } \end{array}$ </td><td>θ</td><td>π</td><td>2,30</td></tr><tr><td colspan="7"> $\mathbf { p } _ { 0 } ^ { \top } V = 0 , r = \| V \| _ { 2 } , \theta = \operatorname { a r c c o s } ( \mathbf { p } _ { 0 } ^ { \top } \mathbf { p } _ { 1 } ) .$ </td></tr><tr><td>Flat torus</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathbf { p } _ { 0 } , \mathbf { p } _ { 1 } \in \mathbb { T } ^ { n }$ </td><td> $V ^ { \top } W$ </td><td> $[ \mathbf { p } _ { 0 } + V ]$ </td><td> $\mathrm { w r a p } ( \mathbf { p } _ { 1 } - \mathbf { p } _ { 0 } )$ </td><td> $\| \operatorname { w r a p } ( \mathbf { p } _ { 1 } - \mathbf { p } _ { 0 } ) \| _ { 2 }$ </td><td>π</td><td>2,30</td></tr><tr><td colspan="7"> $\mathbb { T } ^ { n } = \mathbb { R } ^ { n } / ( 2 \pi \mathbb { Z } ) ^ { n }$  . [·] denotes equivalence modulo  $( 2 \pi \mathbb { Z } ) ^ { n } ;$  wrap selects the representative in  $[ - \pi , \pi ) ^ { n }$ </td></tr><tr><td>Hyperbolic</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathbf { p } _ { 0 } , \mathbf { p } _ { 1 } \in \mathbb { H } ^ { n }$   $\mathbb { H } ^ { n } = \{ \mathbf { p } _ { 0 } \in \mathbb { R } ^ { n + 1 } : \langle \mathbf { p } _ { 0 } , \mathbf { p } _ { 0 } \rangle _ { L } = - 1 , p _ { 0 } ^ { n + 1 } > 0 \} .$ </td><td> $\langle V , W \rangle _ { L }$ </td><td></td><td> $\begin{array} { r l } { \cosh ( r ) \mathbf { p } _ { 0 } + \frac { \sinh r } { r } V } & { { } \frac { \theta } { \sinh \theta } \bigl ( \mathbf { p } _ { 1 } - \cosh ( \theta ) \mathbf { p } _ { 0 } \bigr ) } \end{array}$ </td><td>θ</td><td>∞</td><td>2,30</td></tr><tr><td colspan="7"> $\begin{array} { r } { \langle V , W \rangle _ { L } = \sum _ { i = 1 } ^ { n } V ^ { i } W ^ { i } - V ^ { n + 1 } W ^ { n + 1 } . } \end{array}$   $\mathrm { T a n g e n t s ~ s a t i s f y } \langle \mathbf { p } _ { 0 } , V \rangle _ { L } = 0 ; r = \sqrt { \langle V , V \rangle _ { L } } , \theta = \mathrm { a r c c o s h } ( - \langle \mathbf { p } _ { 0 } , \mathbf { p } _ { 1 } \rangle _ { L } ) .$ </td></tr><tr><td>SPD matrices</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="7"> $\mathbf { p } _ { 0 } , \mathbf { p } _ { 1 } \in \mathrm { S P D } ( p )$   $\operatorname { t r } ( \mathbf { p } _ { 0 } ^ { - 1 } V \mathbf { p } _ { 0 } ^ { - 1 } W )$   $\mathbf { p } _ { 0 } ^ { 1 / 2 } \exp ( A ) \mathbf { p } _ { 0 } ^ { 1 / 2 }$   ${ \bf p } _ { 0 } ^ { 1 / 2 } \log ( B ) { \bf p } _ { 0 } ^ { 1 / 2 }$   $\mathbf { p } _ { 0 } = \mathbf { p } _ { 0 } ^ { \top } \succ 0 ,$  with symmetric tangents  $V , W ;$  dimension  $p ( p + 1 ) / 2 .$  We use  $p = 2 , 7 .$ </td></tr><tr><td> $A = { \bf p } _ { 0 } ^ { - 1 / 2 } V { \bf p } _ { 0 } ^ { - 1 / 2 } , B = { \bf p } _ { 0 } ^ { - 1 / 2 } { \bf p } _ { 1 } { \bf p } _ { 0 } ^ { - 1 / 2 } .$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="7">Grassmannian  $[ \mathbf { p } _ { 0 } ] , [ \mathbf { p } _ { 1 } ] \in \mathrm { G r } ( k , p ) \qquad \mathrm { t r } ( V ^ { \top } W )$   $\exp ( \Omega ) \mathbf { p } _ { 0 }$ </td></tr><tr><td> $\mathbf { p } _ { 0 } \in \mathbb { R } ^ { p \times k }$  represents a subspace, po</td><td></td><td></td><td> $A \arctan ( \Sigma ) B ^ { \top }$ </td><td> $\| \arctan ( \Sigma ) \| _ { F }$ </td><td> $\pi / 2$ </td><td>4,30</td></tr><tr><td colspan="7"> $\begin{array} { r } { \mathbf { \nabla } [ \mathbf { p } _ { 0 } = I _ { k } , \mathbf { p } _ { 0 } ^ { \top } V = 0 ; } \end{array}$  dimension  $k ( p - k )$  . We use  $( k , p ) = ( 2 , 4 ) , ( 5 , 1 1 )$   $\Omega = V \mathbf { p } _ { 0 } ^ { \top } - \mathbf { p } _ { 0 } V ^ { \top } ,$  thin  $\mathrm { S V D } A \Sigma B ^ { \top } = ( I _ { p } - \mathbf { p } _ { 0 } \mathbf { p } _ { 0 } ^ { \top } ) \mathbf { p } _ { 1 } ( \mathbf { p } _ { 0 } ^ { \top } \mathbf { p } _ { 1 } ) ^ { - 1 } ,$  with arctan acting on the diagonal of Σ.</td></tr><tr><td>Rotation group</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="7"> $\mathbf { p } _ { 0 } , \mathbf { p } _ { 1 } \in \mathrm { S O } ( n )$   $\mathrm { t r } ( V ^ { \top } W )$ </td></tr><tr><td>p0</td><td></td><td> $\mathbf { p } _ { 0 } \exp ( \mathbf { p } _ { 0 } ^ { \top } V )$  We use</td><td> $\mathbf { p } _ { 0 } \log ( \mathbf { p } _ { 0 } ^ { \top } \mathbf { p } _ { 1 } )$ </td><td> $\| \log ( \mathbf { p } _ { 0 } ^ { \top } \mathbf { p } _ { 1 } ) \| _ { F }$ </td><td></td><td>3,28</td></tr><tr><td colspan="7"> $\mathbf { p } _ { 0 } = I _ { n } ,$  det  $\mathbf { p } _ { 0 } = 1 ;$  dimension  $n ( n - 1 ) / 2$   $n = 3 , 8 .$  Tangents have the form  $\begin{array} { r } { V = \mathbf { p } _ { 0 } K , K ^ { \top } = - K . } \end{array}$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

where $\rho ( { \bf x } ) = q ( { \bf x } ) ^ { - 1 / 2 }$ is the bandwidth function and q is a kernel density estimate. The bandwidth of $q$ at each point is the root-mean-square distance to the point’s seven nearest neighbors. We choose the scale ϵ and intrinsic dimension estimate <sup>ˆ</sup>d jointly following the procedure of Jones and Lanners (2026). For each $\epsilon \in \{ 2 ^ { - 1 0 } , 2 ^ { - 9 . 7 5 } , \hdots , 2 ^ { 9 . 7 5 } \}$ , we average the kernel weights and compute the slope of this average on a log-log scale. We select the value of ϵ with the largest slope and set $\hat { d }$ to twice that slope. Since the local bandwidths depend on the sampling density, we first use this procedure to estimate the density, update the bandwidths, and then repeat it to obtain the values of ϵ and <sup>ˆ</sup>d used in $\tilde { \bf K } .$ . The symmetrized row sums of P, normalized to sum to one, form the measure µ over the cloud.

We compute the leading, $n _ { 0 } = 2 4 0$ for the disk and $n _ { 0 } = 1 2 0$ for the other two datasets, eigenfunctions of P and store their values on the cloud in the $m \times n _ { 0 }$ matrix U, which satisfies $\mathbf { U } ^ { T } \mathrm { d i a g } ( \pmb { \mu } ) \mathbf { U } = \mathbf { I } _ { n _ { 0 } }$ . The eigenfunctions are the smoothest functions on the cloud, so replacing the values $\mathbf { x } _ { c } \in \mathbb { R } ^ { m }$ of each coordinate function by $\mathbf { U U } ^ { T } \operatorname { d i a g } ( \pmb { \mu } ) \mathbf { x } _ { c }$ removes its noisiest components (Jones and Lanners, 2026, Computational Note 7).

The geometry corresponding to the point cloud is given through the carré du champ of the smoothed coordinates. It is a covariance under one step of the Markov chain (Jones and Lanners, 2026, Equation 6) given by,

$$
\Gamma _ { j } ( x _ { c } , x _ { c ^ { \prime } } ) = \frac { 2 } { \epsilon \rho ( { \bf x } ^ { ( j ) } ) ^ { 2 } } \mathrm { C o v } \left[ x _ { c } ( X ) , x _ { c ^ { \prime } } ( X ) : X \sim { \bf P } _ { j } \right] .\tag{37}
$$

We average the per-point tensors once under $\mathbf { P } ,$ , symmetrize them, and divide all of them by the median over the cloud of the largest eigenvalue.

Finally, the cometric is defined as a soft, Gaussian weighted average of the CDC over the point cloud

$$
\begin{array} { l } { { \displaystyle g _ { x } ( \mathbf { x } ) ^ { - 1 } = \delta I _ { n } + ( 1 - \delta ) a ( \mathbf { x } ) \sum _ { j = 1 } ^ { m } w ^ { ( j ) } ( \mathbf { x } ) \Gamma _ { j } ( \mathbf { x } , \mathbf { x } ) } , } \\ { { \displaystyle w ^ { ( j ) } ( \mathbf { x } ) = \frac { \exp \big ( - \| \mathbf { x } - \mathbf { x } ^ { ( j ) } \| _ { 2 } ^ { 2 } / 2 \sigma ^ { 2 } \big ) } { \sum _ { l = 1 } ^ { m } \exp \big ( - \| \mathbf { x } - \mathbf { x } ^ { ( l ) } \| _ { 2 } ^ { 2 } / 2 \sigma ^ { 2 } \big ) } } } \end{array}\tag{38}
$$

where $\delta = 2 \times 1 0 ^ { - 3 }$ and σ is 0.08 for the disk and the median distance to the eigth neighbor for the two moons and nested circles datasets. $a ( \mathbf { x } )$ is an extra factor to have the geometry behave well away from the support of the data. For two moons and nested circles, $a ( \mathbf { x } ) = 1$ and for the disk,

$$
a ( \mathbf { x } ) = \frac { S ( \mathbf { x } ) } { S ( \mathbf { x } ) + \tau } , \qquad S ( \mathbf { x } ) = \sum _ { j = 1 } ^ { m } \exp \left( - \frac { \| \mathbf { x } - \mathbf { x } ^ { ( j ) } \| _ { 2 } ^ { 2 } } { 2 \sigma _ { s } ^ { 2 } } \right) ,\tag{39}
$$

with $( \sigma _ { s } , \tau ) = ( 0 . 0 2 , 1 )$

Chart fitting for the diffusion illustration For Figure 2, we fit one chart per displayed base point to its 4,096 nearest neighbors within the same connected component, excluding the base point. Each chart has four affine coupling layers with dense linear mixing, two gated conditioner blocks of width 32, and a one-block base point encoder of width 16 with 8 output features. We use metric preconditioning and center inputs at the base point, scaling by the median neighbor distance measured with $g _ { x } ( \mathbf { x } _ { 0 } )$ . We minimize the normal loss with Adam at a constant learning rate of $3 \times 1 0 ^ { - 4 }$ and unit gradient clipping, using FP64 chart parameters and batches of 64 endpoints (128 for the disk). Let $r _ { 0 }$ be the median encoded neighbor norm at initialization. We train for 5,000 steps with $\gamma = 0 . 1 r _ { 0 } / \sqrt { 2 } .$ then restart Adam from the best checkpoint with $\gamma = 0 . 5 r _ { 0 }$ for another 2,000 steps on moons and circles or 5,000 on the disk. Every 100 steps we evaluate fixed independent bridge draws on the same training endpoints and retain the lowest-loss checkpoint for the current $\gamma$ . The cloud seed is 0 and the three chart initialization seeds are 0, 1, 2.

Normalizing-flow pullback illustration For the pullback panel of Figure 2, we fit an unconditional flow h to the uneven-concentration dataset by maximum likelihood under a standard Gaussian prior, using the four-layer affine coupling architecture described above. We compose its encoder with the componentwise standard Gaussian CDF, $H = \Phi \circ h ,$ , and define the pullback metric $g _ { x } = D H ^ { \top } D H .$ whose volume density equals the fitted data density. We freeze this metric and fit one flow chart at each of the same three base points, using the diffusion illustration’s chart architecture and training procedure with FP32.

For both geometries, we decode coordinate grids restricted to disks of a common radius within each panel, orient the grids using rotation-only ICA on encoded neighbors, and overlay the same 600 observations. The backgrounds use separate logarithmic color scales: blue shows inverse volume density for the diffusion metric, and green shows volume density for the pullback metric. Panels (b) and (d) show normalized histograms of the learned coordinate norms of each base point’s 4,096 nearest neighbors, using 40 equally spaced bins shared across base points within each geometry. Colors identify corresponding base points across the geometry panels and histograms.

## C.4 Riemannian metric matching

For higher dimensional datasets, we use Riemannian metric matching (RMM) (Bamberger et al., 2026) to learn its geometry. Let $p _ { \varepsilon | 0 } ( \mathbf { x } _ { \varepsilon } \mid \mathbf { x } ) = \mathcal { N } ( \mathbf { x } _ { \varepsilon } \mid \mathbf { x } , \varepsilon I _ { n } )$ denote the transition distribution at time ε of Brownian motion started at x, let $p _ { 0 , \varepsilon } ( \mathbf { x } , \mathbf { x } _ { \varepsilon } ) = p _ { \mathrm { d a t a } } ( \mathbf { x } ) p _ { \varepsilon | 0 } ( \mathbf { x } _ { \varepsilon } \mid \mathbf { x } )$ denote the joint distribution of a data point and its noised version, let $p _ { \varepsilon } ( \mathbf { x } _ { \varepsilon } )$ denote the marginal of the noised point, and let $p _ { 0 | \varepsilon } ( \cdot | \mathbf { x } _ { \varepsilon } )$ denote the posterior over the clean point given the noised one. The posterior mean is the denoiser

$$
\bar { \mathbf { x } } _ { 0 } ( \mathbf { x } _ { \varepsilon } ) = \mathbb { E } _ { \mathbf { x } _ { 0 } \sim p _ { 0 | \varepsilon } ( \mathbf { \cdot } | \mathbf { x } _ { \varepsilon } ) } \left[ \mathbf { x } _ { 0 } \right]\tag{40}
$$

and the geometry is the family of cometrics, indexed by ε, given by the scaled posterior covariance

$$
g _ { \varepsilon , x } ( \mathbf { x } _ { \varepsilon } ) ^ { - 1 } = \frac { 1 } { 2 \varepsilon } \operatorname { C o v } _ { \mathbf { x } _ { 0 } \sim p _ { 0 | \varepsilon } ( \cdot | \mathbf { x } _ { \varepsilon } ) } \left[ \mathbf { x } _ { 0 } \right] = \frac { 1 } { 2 } D \bar { \mathbf { x } } _ { 0 } ( \mathbf { x } _ { \varepsilon } )\tag{41}
$$

where the second equality is Tweedie’s formula for the posterior covariance (Efron, 2011). This is the mean centered carré du champ of RMM. At a noised point $\mathbf { x } _ { \varepsilon }$ the covariance spans the directions along which clean data that could have produced $\mathbf { x } _ { \varepsilon }$ vary, so the cometric is aligned with the data manifold. RMM fits a parametric cometric, $g _ { \varepsilon , x } ^ { \theta }$ , to the target by minimizing the following regression:

$$
\begin{array} { r l } & { \mathcal { L } ( \theta ) = \underset { \varepsilon \sim p ( \varepsilon ) } { \mathbb { E } } \underset { \mathbf { x } _ { \varepsilon } \sim p _ { \varepsilon } } { \mathbb { E } } \left[ \left. g _ { \varepsilon , x } ^ { \theta } ( \mathbf { x } _ { \varepsilon } ) ^ { - 1 } - g _ { \varepsilon , x } ( \mathbf { x } _ { \varepsilon } ) ^ { - 1 } \right. _ { F } ^ { 2 } \right] } \\ & { \quad \quad = \underset { \varepsilon \sim p ( \varepsilon ) } { \mathbb { E } } \underset { \mathbf { x } _ { \varepsilon } \sim \sim p _ { 0 , \varepsilon } } { \mathbb { E } } \left[ \left. g _ { \varepsilon , x } ^ { \theta } ( \mathbf { x } _ { \varepsilon } ) ^ { - 1 } - \frac { 1 } { 2 \varepsilon } \left( \mathbf { x } - \bar { \mathbf { x } } _ { 0 } ( \mathbf { x } _ { \varepsilon } ) \right) ( \mathbf { x } - \bar { \mathbf { x } } _ { 0 } ( \mathbf { x } _ { \varepsilon } ) ) ^ { \top } \right. _ { F } ^ { 2 } \right] + \mathrm { c o n s t . } } \end{array}\tag{42}
$$

The second expression is tractable to optimize provided that we can compute the denoiser. For this, we either pretrain a denoiser, or download a pretrained denoiser from the internet, which we denote by $D _ { \varepsilon } ( \mathbf x _ { \varepsilon } ) \approx \bar { \mathbf x } _ { 0 } ( \mathbf x _ { \varepsilon } )$

## C.4.1 Low rank parameterization

We parameterize the cometric using the approximately low rank parameterization of RMM:

$$
g _ { \varepsilon , x } ^ { \theta } ( \mathbf { x } _ { \varepsilon } ) ^ { - 1 } = \delta I _ { n } + ( 1 - \delta ) M _ { \varepsilon } ^ { \theta } ( \mathbf { x } _ { \varepsilon } ) ^ { \top } M _ { \varepsilon } ^ { \theta } ( \mathbf { x } _ { \varepsilon } )\tag{43}
$$

where $M _ { \varepsilon } ^ { \theta } ( \mathbf { x } _ { \varepsilon } ) \in \mathbb { R } ^ { d \times n }$ is the unconstrained output of a neural network, and $d \ll n$

We train with δ incorporated into the objective, which yields

$$
\begin{array} { r l r } & { } & { \mathcal { L } ( \theta ) = \underset { \varepsilon \sim p ( \varepsilon ) ( \mathbf { x } , \mathbf { x } _ { \varepsilon } ) \sim p _ { 0 , \varepsilon } } { \mathbb { E } } \Big [ ( 1 - \delta ) ^ { 2 } \| M _ { \varepsilon } ^ { \theta } ( \mathbf { x } _ { \varepsilon } ) M _ { \varepsilon } ^ { \theta } ( \mathbf { x } _ { \varepsilon } ) ^ { \top } \| _ { F } ^ { 2 } + 2 \delta ( 1 - \delta ) \| M _ { \varepsilon } ^ { \theta } ( \mathbf { x } _ { \varepsilon } ) \| _ { F } ^ { 2 } } \\ & { } & { \qquad - \frac { 1 - \delta } { \varepsilon } \big \| M _ { \varepsilon } ^ { \theta } ( \mathbf { x } _ { \varepsilon } ) \big ( \mathbf { x } - D _ { \varepsilon } ( \mathbf { x } _ { \varepsilon } ) \big ) \big \| _ { 2 } ^ { 2 } \Big ] + \mathrm { c o n s t . } } \end{array}\tag{44}
$$

## C.4.2 Denoiser architecture

We train a denoiser for alanine dipeptide. The denoiser is parameterized by a residual network, and predicts the clean point under additive Gaussian corruption, so its noise argument is the variance ε used by RMM. The denoiser uses the parameterization of Karras et al. (2022), where $D _ { \phi } ( \mathbf { x } _ { \varepsilon } , \varepsilon ) =$ $c _ { \mathrm { s k i p } } ( \varepsilon ) \mathbf { x } _ { \varepsilon } + c _ { \mathrm { o u t } } ( \varepsilon ) F _ { \phi } ( c _ { \mathrm { i n } } ( \varepsilon ) \mathbf { x } _ { \varepsilon } , \varepsilon )$ . Here, $F _ { \phi }$ is a trainable neural network conditioned on the noise variance, and the fixed functions $c _ { \mathrm { i n } } , c _ { \mathrm { o u t } }$ , and $c _ { \mathrm { s k i p } }$ scale its input, its output, and the skip connection, respectively. We set the reference standard deviation of the coordinates to $\sigma _ { \mathrm { { d a t a } } } = 1$ , since every dataset is standardized to unit root mean square spread, giving

$$
c _ { \mathrm { i n } } ( \varepsilon ) = \frac { 1 } { \sqrt { \varepsilon + 1 } } , \qquad c _ { \mathrm { o u t } } ( \varepsilon ) = \sqrt { \frac { \varepsilon } { \varepsilon + 1 } } , \qquad c _ { \mathrm { s k i p } } ( \varepsilon ) = \frac { 1 } { \varepsilon + 1 }\tag{45}
$$

The network $F _ { \phi }$ is a linear layer from $\mathbb { R } ^ { n }$ to width $w ,$ , followed by L residual blocks of width $w ,$ and a linear layer back to $\mathbb { R } ^ { n }$ whose weights are initialized to zero so that the denoiser starts as the identity map. The embedding for $\boldsymbol { \varepsilon } , \mathbf { e } ( \boldsymbol { \varepsilon } ) \bar { \in } \mathbb { R } ^ { c }$ , is built from 32 Fourier features of log ε at the frequencies $1 0 0 0 ^ { - \displaystyle k / 3 2 }$ for $k = 0 , \ldots , 3 1$ , whose cosines and sines are passed through two linear layers with a SiLU between them. Each residual block is conditioned on the embeddings using FiLM (Perez et al., 2018),

$$
\mathbf { h } \mapsto \mathbf { h } + W _ { 2 } \operatorname { S i L U } \bigl ( ( \mathbf { 1 } + \mathbf { a } ) \odot \operatorname { S i L U } ( W _ { 1 } \mathbf { h } ) + \mathbf { b } \bigr ) , \qquad ( \mathbf { a } , \mathbf { b } ) = W _ { c } \operatorname { S i L U } ( \mathbf { e } ( \varepsilon ) )\tag{46}
$$

where $W _ { 2 }$ is initialized to zero so that each block also starts as the identity. We use $w = 5 1 2 , L = 8$ and $c = 1 2 8$ , which is about 5M parameters. See Table 3 for the denoiser and cometric architectures and training details.

Denoiser training We train the denoiser on the training split with the denoising loss of Karras et al. (2022), which weights the squared error by $c _ { \mathrm { o u t } } ( \varepsilon ) ^ { - 2 }$

$$
\mathcal { L } ( \phi ) = \underset { \varepsilon } { \mathbb { E } } \underset { ( \mathbf { x } , \mathbf { x } _ { \varepsilon } ) \sim p _ { 0 , \varepsilon } } { \mathbb { E } } \left[ c _ { \mathrm { o u t } } ( \varepsilon ) ^ { - 2 } \lVert D _ { \phi } ( \mathbf { x } _ { \varepsilon } , \varepsilon ) - \mathbf { x } \rVert _ { 2 } ^ { 2 } \right]\tag{47}
$$

We draw ε log-uniformly over [0.01, 5]. We train with Adam, clip gradients to unit global norm, and ramp the learning rate linearly over the first 1,000 steps. We train the denoiser for a maximum of 1,000,000 steps. We keep an exponential moving average of the weights with decay 0.999, evaluate its loss on the validation split every 1,000 steps with early stopping enabled, and select the averaged checkpoint with the lowest validation loss. A run stops once 50,000 steps pass without an improvement in the validation loss. We sweep the peak learning rate over $\{ 5 \times \mathrm { i } 0 ^ { - 5 } , 1 0 ^ { - 4 } , 2 \times$ $1 0 ^ { - 4 } , \stackrel { \cdot } { 4 } \times 1 0 ^ { - 4 } , 8 \times 1 0 ^ { - 4 } \}$ with three seeds each, and select the run whose selected checkpoint has the smallest loss on the validation split.

Cometric architecture The cometric shares the exact same architecture as the denoiser, except that the last layer maps to a larger vector of size $\mathbb { R } ^ { d \times n }$ to produce the matrix $M _ { \varepsilon } ^ { \theta }$ , following Section C.4.1. The cometric does not use the coefficients $c _ { \mathrm { i n } } , c _ { \mathrm { o u t } }$ , and $c _ { \mathrm { s k i p } } ,$ so $M _ { \varepsilon } ^ { \theta } ( \mathbf x _ { \varepsilon } )$ is the output of the network evaluated at $\mathbf { x } _ { \varepsilon }$ directly. We use a full rank parameterization with $d = n$ and set $\delta = 1 \times 1 0 ^ { - 4 }$

Cometric training We train the cometric on the training data with the objective from Section C.4.1, centered on the selected denoiser. The optimization procedure is identical to how we trained the denoiser.

## D Experiment details

Here we give the details of our experiments.

Timing details We measure wall clock time for all of the experiments in this paper in the same way. This method takes a compiled JAX program as input, with its inputs already in device memory, so that compilation time and other preprocessing steps are not included in the reported time. The program is run once as a warmup, and then 5 times after, from which we report the median time. All timings are reported on one NVIDIA A16 in float32 and every program is implemented in JAX (Bradbury et al., 2018) and compiled by XLA with its default settings.

## D.1 Logarithmic map experiment details

We generate data on each manifold following the procedure in Section C.1. For each manifold, we generate 2048 basepoints for training the amortized models, 64 for validation and 128 for testing.

Table 3: Denoiser and cometric architectures and training on alanine dipeptide.
<table><tr><td>Property</td><td>Alanine dipeptide</td></tr><tr><td>Residual network architecture Dimension n</td><td>30</td></tr><tr><td>Width w Residual blocks L</td><td>512</td></tr><tr><td>Embedding size c</td><td>8 128</td></tr><tr><td>Total parameters (M)</td><td>5.32</td></tr><tr><td>Denoiser</td><td></td></tr><tr><td>Training points</td><td>525,000</td></tr><tr><td>Validation points</td><td>75,000</td></tr><tr><td>Test points</td><td></td></tr><tr><td>Noise range</td><td>150,000</td></tr><tr><td>Batch size</td><td>[0.01, 5]</td></tr><tr><td>Maximum training steps</td><td>512</td></tr><tr><td></td><td>1,000,000</td></tr><tr><td>EMA decay</td><td>0.999</td></tr><tr><td>Selected learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Selected step</td><td>245,000</td></tr><tr><td>Validation loss</td><td></td></tr><tr><td>Test loss</td><td>6.336</td></tr><tr><td></td><td>6.361</td></tr><tr><td>Cometric</td><td></td></tr><tr><td>Rank d</td><td>30</td></tr><tr><td>Floor δ</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Total parameters (M)</td><td></td></tr><tr><td></td><td> $5 . 7 6$ </td></tr><tr><td>Selected learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Selected step</td><td>251,000</td></tr><tr><td>Validation loss</td><td>-0.7793</td></tr><tr><td>Test loss</td><td>-0.7762</td></tr></table>

For each base point, we sample 1024 associated endpoints to train the MFM and flow chart with. These methods are then evaluated on endpoints of the 128 test base points. All methods use the same test endpoints. The closed form manifolds provide exact references. The data geometries use the reference solutions described in Section D.2. After reference filtering, the two moons has queries at 124 test base points and the other geometries have queries at all 128. We report the mean of $\mathcal { R } _ { \mathrm { l o g m a p } } ( \mathbf { x } _ { 0 } )$ and its standard error across the test base points, separately for each manifold and dimension.

## D.1.1 Methods

Our baselines employ geodesic interpolation to retrieve normal coordinate and logmap estimates. These methods parameterize a curve $\gamma _ { ( \cdot ) } ^ { \phi } \colon [ 0 , 1 ] \to { \mathcal { M } } _ { x }$ that starts at a base point $\mathbf { x } _ { \mathrm { 0 } }$ and ends at a query point x, and approximate a geodesic by minimizing its energy or solving the geodesic equation. The iterative algorithms (GP and GEORCE) are run until the relative decrease of their energy between iterations falls below a tolerance $\epsilon _ { \mathrm { s t o p } } = 1 \times 1 0 ^ { - 5 }$ , and are all parameterized and optimized in float32. Each solver is stopped at 1,000 max iterations, and fails if its solution is nonfinite or the solver reaches this cap without meeting its stopping criterion. Failed queries are left out of both the wall clock time and the errors.

GP We implement the fixed-point solver of Arvanitidis et al. (2019), which represents a curve by a Gaussian-process posterior mean conditioned on its endpoints and K acceleration vectors at uniformly spaced times between 0 and 1, with the endpoints included. We use a straight-line prior mean, a squared-exponential kernel with length scale $\lambda = \sqrt { 1 / ( 2 ( K - 1 ) ) }$ , and for numerical stability in FP32, acceleration regularization of $1 0 ^ { - 2 }$ . The model iteratively infers the acceleration of the posterior mean of a Gaussian process, that is conditioned on the endpoint conditions and the accelerations.

Table 4: Sizes of the amortized models and the wall clock cost of computing 1024 normal coordinate estimates at a fixed base point.
<table><tr><td rowspan="2">n</td><td colspan="2">Flow chart</td><td colspan="3">MFM, parameter matched</td><td colspan="3">MFM, time matched</td></tr><tr><td>Parameters</td><td>µs</td><td>Width × blocks</td><td>Parameters</td><td>µs</td><td>Width × blocks</td><td>Parameters</td><td>µs</td></tr><tr><td>2</td><td>187,504</td><td>0.98</td><td> $6 4 \times 1 5$ </td><td>188,674</td><td>1.51</td><td> $6 4 \times 9$ </td><td>113,410</td><td>1.01</td></tr><tr><td>3</td><td>191,296</td><td>1.00</td><td> $6 4 \times 1 5$ </td><td>188,867</td><td>1.71</td><td> $6 4 \times 6$ </td><td>75,971</td><td>0.97</td></tr><tr><td>4</td><td>195,104</td><td>1.05</td><td> $9 6 \times 7$ </td><td>197,572</td><td>1.39</td><td> $6 4 \times 6$ </td><td>76,164</td><td>1.01</td></tr><tr><td>28</td><td>291,296</td><td>1.33</td><td> $9 6 \times 1 0$ </td><td>288,604</td><td>2.05</td><td> $6 4 \times 6$ </td><td>80,796</td><td>1.32</td></tr><tr><td>30</td><td>299,728</td><td>1.35</td><td> $1 2 8 \times 6$ </td><td>309,790</td><td>1.58</td><td> $6 4 \times 9$ </td><td>118,814</td><td>1.30</td></tr></table>

GEORCE We also use the GEORCE (Rygaard and Hauberg, 2026) algorithm to solve for a geodesic interpolation. This approach parameterizes the geodesic with K knot points, but uses an iterative algorithm to solve for the knot points without assuming any particular curve parameterization. GEORCE recovers the geodesic interpolation by solving the discretized energy as a control problem with boundary constraints. GEORCE returns $\dot { K }$ knot points rather than a curve, so we interpolate its knots with the piecewise cubic Hermite interpolant with backward differences of Morrill et al. (2021) for evaluation. The initial velocity of GEORCE is the forward difference $( K + 1 ) ( \mathbf { y } _ { 1 } - \mathbf { x } _ { 0 } )$ , where $\mathbf { y } _ { 0 } = \mathbf { x } _ { 0 } , \mathbf { y } _ { 1 } , \dots , \mathbf { y } _ { K } , \mathbf { y } _ { K + 1 } = \mathbf { x }$ at the times $t _ { k } = k / ( K + 1 )$ are the knots that are solved for with the algorithm.

Metric flow matching (MFM) Our neural network baseline is MFM (Kapusniak et al., 2024). This´ method parameterizes each curve as follows:

$$
\mathbf { x } _ { \mathrm { c u r v e } , t } ^ { \theta } = ( 1 - t ) \mathbf { x } _ { 0 } + t \mathbf { x } _ { 1 } + t ( 1 - t ) \mathrm { N N } _ { \theta } ( t , \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } )\tag{48}
$$

where $\mathrm { N N } _ { \theta }$ is a neural network. We construct $\mathrm { N N } _ { \theta }$ as a gated residual network. MFM is trained to minimize the energy E. The training procedure is identical to the flow chart’s, below.

We choose the number of blocks and hidden size separately for two MFM baselines. The first has a parameter count modestly close to that of the four-layer flow chart, and the second has a comparable time to compute the fast normal coordinate estimates $\mathbf { z } ^ { ( i ) }$ of Section D.1.2 for a batch of 1024 queries. For the flow chart, this is $h _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { x } ^ { ( i ) } )$ , and for MFM it is $\mathbf { J } _ { x } ^ { - 1 } \dot { \mathbf { x } } _ { 0 } ^ { ( i ) }$ . Table 4 lists the model sizes and computation times. Times are microseconds per query on an NVIDIA A16 in FP32.

Flow chart The flow chart used for the amortized MFM comparison and the preconditioner is a conditional neural spline flow (Durkan et al., 2019) with 4 rational quadratic spline coupling layers, each with 8 bins on [−10, 10] in both low and high dimensions, that is initialized to be close to the identity transformation. Before each coupling layer, we place a parametric elementwise scale and shift that is conditioned on the base point encoder and a dense invertible linear map (with an LU parameterization). Each coupling layer parameterizes its spline using a FiLM conditioned (Perez et al., 2018) residual network with three residual blocks of width 32. The conditioning is computed with a separate residual network that encodes the base point into a 64-dimensional feature vector for all coupling layers with 8 gated residual blocks of width 64. The fixed-base-point and runtime experiments use the wider coupling networks described in Section D.3. We train our method with the normal loss in Proposition 3 under the sampling distribution of Section 4.2, with $\gamma = 0 . 1 r _ { \mathrm { m a x } } / \sqrt { n } .$

We minimize this loss with Adam in single precision at a learning rate of $3 \times 1 0 ^ { - 4 }$ with linear warmup over the first 100 steps. Each training batch holds 16 base points with 32 endpoints each, for a total of 512 pairs, so the base point encoder is evaluated 16 times per step and the coupling layers on every pair. For each pair we draw one time $t \sim \mathrm { U } ( 0 , 1 )$ and $\mathbf { z } _ { t }$ according to the sampling distribution in Section 4.2. Gradient updates are clipped to have unit norm. We keep an exponential moving average of the parameters with decay 0.999, and this average is the model we evaluate and report. The validation loss is evaluated on held out pairs every 500 steps. The learning rate is constant until step 1,000,000 and then decays linearly to zero at step 2,000,000, where training ends, and we keep the average with the lowest validation loss.

## D.1.2 Evaluation metrics

Recall that each method we test takes as input a base point $\mathbf { x } _ { \mathrm { 0 } }$ , and query points, $\{ \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( N ) } \}$ , and produces a set of coordinates, $\{ \mathbf { z } ^ { ( 1 ) } , \ldots , \mathbf { z } ^ { ( N ) } \}$ }, and tangent vectors, $\{ U _ { x } ^ { ( 1 ) } , \ldots , U _ { x } ^ { ( N ) } \}$ , representing the normal coordinates and log map solutions. In this section, we review how each method produces the coordinates and tangent vector solutions. The baseline methods, and our method, offer a curve $\mathbf { x } _ { \cdot } ^ { ( i ) }$ from the base point to a query point $\mathbf { x } ^ { ( i ) }$ at $t = 1$ , that after training approximates a geodesic. We use each of these curves to construct a fast logmap and normal coordinate solution.

Method solutions The fast logmap estimate is simply the initial velocity of the curve, and the corresponding normal coordinate is the logmap scaled by a g-orthogonal matrix. Let $\mathbf { J } _ { x } = g _ { x } ( \mathbf { x } _ { 0 } ) ^ { - \frac { 1 } { 2 } }$ be this matrix. Then the fast logmap and normal coordinate solutions corresponding to a method applied to base point $\mathbf { x } _ { \mathrm { 0 } }$ and query $\mathbf { x } ^ { ( i ) }$ are

$$
U _ { x } ^ { ( i ) } = \dot { \mathbf { x } } _ { 0 } ^ { ( i ) } , \qquad \mathbf { z } ^ { ( i ) } = \mathbf { J } _ { x } ^ { - 1 } \dot { \mathbf { x } } _ { 0 } ^ { ( i ) }\tag{49}
$$

For the flow chart, whose curve is $\mathbf { x } _ { t } ^ { ( i ) } = f _ { \mathbf { x } _ { 0 } } ^ { \theta } ( t h _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { x } ^ { ( i ) } ) )$ , the fast normal coordinate is the chart’s own output, $\mathbf { z } ^ { ( i ) } = h _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { x } ^ { ( i ) } )$ , and the fast logmap is $U _ { x } ^ { ( i ) } = D f _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { 0 } ) \mathbf { z } ^ { ( i ) }$ as described in Section 3.

The GP and GEORCE methods fail if the solution is not finite, or the solver stops before meeting its stopping criterion within the maximum number of iterations. We write $\mathcal { T } ( \mathbf { x } _ { 0 } ) \subseteq \{ 1 , \dots , N \}$ to denote the indices of the queries of base point $\mathbf { x } _ { \mathrm { 0 } }$ at which the method does not fail, and $\mathcal { I } ( \mathbf { x } _ { \mathrm { 0 } } )$ for the indices of queries with valid reference solutions. MFM and the flow chart fail at a query only if its solution is not finite. Each method is graded on its own successful queries. The failed fraction is below 0.72% for each numerical logmap setting, and no evaluated chart logmap query fails.

Evaluation metrics We evaluate the solutions against the ground truth coordinates and logarithmic maps, $\{ \mathbf { s } ^ { ( 1 ) } , \ldots , \mathbf { s } ^ { ( N ) } \}$ and $\{ V _ { x } ^ { ( 1 ) } , \ldots , V _ { x } ^ { ( N ) } \}$ .

Wallclock Since we are concerned with speed, we time how long it takes (in wall clock time) to compute the solutions, as measured by Section D. We let $\tau ( \mathbf { x } _ { 0 } )$ denote the wall clock time to compute the solutions $\mathbf { z } ^ { ( i ) }$ and $U _ { x } ^ { \left( i \right) }$ for every $i \in \mathcal { T } ( \mathbf { x } _ { 0 } )$ , divided by $| \mathcal { T } ( \mathbf { x } _ { 0 } )$ |, so that $\tau ( \mathbf { x } _ { 0 } )$ is a time per query. Since $\mathcal { T } ( \mathbf { x } _ { \mathrm { 0 } } )$ is not known in advance, we first run the method on all N queries of $\mathbf { x } _ { \mathrm { 0 } }$ to find it, and then time a second run on the queries in $\mathcal { T } ( \mathbf { x } _ { \mathrm { 0 } } )$ alone.

For the GEORCE and GP baselines, $\tau ( \mathbf { x } _ { 0 } )$ includes the time to solve for the curve. $\tau ( \mathbf { x } _ { 0 } )$ does not include the time to pretrain the flow chart and MFM on the training set. For the flow chart, the time to compute the base point encoding is included in the time.

Because a batch of queries iterates until its slowest curve stops, we sweep over batch sizes in {64, 128, 256, 512, 1024} for each of the GP and GEORCE, at each knot count and geometry on the first 16 test base points and take the batch size with the smallest mean of $\tau ( \mathbf { x } _ { 0 } )$ over these base points. The flow chart and both MFMs are evaluated at a batch size of 1024. On this best batch size, we record the time and peak memory usage of every test base point. Table 5 lists the batch sizes selected for the GP and GEORCE.

We report the mean of $\tau ( \mathbf { x } _ { 0 } )$ over the test base points.

Logmap To evaluate the logmap solutions, we compute the normalized RMSE $\mathcal { R } _ { \mathrm { l o g m a p } } ( \mathbf { x } _ { 0 } )$ defined in Section 5.1. We compute the root mean squared g-norm error between the logmap solution and the ground truth

$$
\mathcal { E } _ { \mathrm { l o g m a p } } ( \mathbf { x } _ { 0 } ) = \left( \frac { 1 } { \left| \mathcal { T } ( \mathbf { x } _ { 0 } ) \right| } \sum _ { i \in \mathcal { I } ( \mathbf { x } _ { 0 } ) } \left\| \boldsymbol { U } _ { x } ^ { ( i ) } - \boldsymbol { V } _ { x } ^ { ( i ) } \right\| _ { g } ^ { 2 } \right) ^ { \frac { 1 } { 2 } }\tag{50}
$$

We divide $\mathcal { E } _ { \mathrm { l o g m a p } } ( \mathbf { x } _ { 0 } )$ by the reference scale $r _ { \mathrm { r e f } } ( \mathbf { x } _ { 0 } )$ , which uses all queries in $\mathcal { I } ( \mathbf { x } _ { 0 } )$ and is shared across methods. We average $\mathcal { R } _ { \mathrm { l o g m a p } } ( \mathbf { x } _ { 0 } )$ ) over base points with valid readings and report its mean and standard error multiplied by 100.

Table 5: Batch sizes selected for the GP and GEORCE at each knot count K, the size with the smallest mean of $\tau ( \mathbf { x } _ { 0 } )$ over the first 16 test base points.
<table><tr><td></td><td colspan="5"></td><td colspan="5">GEORCE</td></tr><tr><td>Manifold</td><td>n</td><td> $K = 4$ </td><td>8</td><td>16</td><td>32</td><td> $K = 4$ </td><td></td><td>8</td><td>16</td><td>32</td></tr><tr><td>Sphere</td><td>2</td><td>1024</td><td>1024</td><td>1024</td><td>1024</td><td>1024</td><td>1024</td><td></td><td>1024</td><td>512</td></tr><tr><td>Flat torus</td><td>2</td><td>1024</td><td>1024</td><td>1024</td><td>1024</td><td></td><td>1024</td><td>1024</td><td>1024</td><td>1024</td></tr><tr><td>Hyperbolic</td><td>2</td><td>1024</td><td>1024</td><td>1024</td><td></td><td>512</td><td>1024</td><td>1024</td><td>1024</td><td>1024</td></tr><tr><td>SPD</td><td>3</td><td>1024</td><td>1024</td><td>1024</td><td>512</td><td></td><td>1024</td><td>512</td><td>256</td><td>128</td></tr><tr><td>Grassmann</td><td>4</td><td>512</td><td>128</td><td>64</td><td>64</td><td></td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Rotations</td><td>3</td><td>1024</td><td>1024</td><td>1024</td><td></td><td>64</td><td>256</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Sphere</td><td>30</td><td>64</td><td>64</td><td>64</td><td>64</td><td></td><td>64</td><td>128</td><td>64</td><td>64</td></tr><tr><td>Flat torus</td><td>30</td><td>64</td><td>128</td><td>64</td><td></td><td>64</td><td>64</td><td>128</td><td>128</td><td>64</td></tr><tr><td>Hyperbolic</td><td>30</td><td>64</td><td>64</td><td>64</td><td></td><td>64</td><td>64</td><td>128</td><td>64</td><td>64</td></tr><tr><td>SPD</td><td>28</td><td>128</td><td>64</td><td>64</td><td></td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Grassmann</td><td>30</td><td>64</td><td>64</td><td>64</td><td></td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Rotations</td><td>28</td><td>128</td><td>64</td><td></td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td></tr></table>

Table 6: Number of test queries that fail for the GP and GEORCE at each knot count K, out of 131,072 queries per geometry, 1,024 at each of the 128 test base points.
<table><tr><td></td><td></td><td colspan="4">GP</td><td colspan="4">GEORCE</td></tr><tr><td>Manifold</td><td>n</td><td> $K = 4$ </td><td>8</td><td>16</td><td>32</td><td> $K = 4$ </td><td>8</td><td>16</td><td>32</td></tr><tr><td>Sphere</td><td>2</td><td>246</td><td>32</td><td>3</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Flat torus</td><td>2</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Hyperbolic</td><td>2</td><td>0</td><td>0</td><td>0</td><td>0</td><td></td><td>0 0</td><td>0</td><td>0</td></tr><tr><td>SPD</td><td>3</td><td>0</td><td>0</td><td>0</td><td>0</td><td></td><td>0 0</td><td>0</td><td>0</td></tr><tr><td>Grassmann</td><td>4</td><td>1</td><td>0</td><td>0</td><td>0</td><td></td><td>0 0</td><td>0</td><td>0</td></tr><tr><td>Rotations</td><td>3</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Sphere</td><td>30</td><td>592</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Flat torus</td><td>30</td><td>18</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Hyperbolic</td><td>30</td><td>151</td><td>0</td><td>0</td><td>0</td><td></td><td>0 0</td><td>0</td><td>0</td></tr><tr><td>SPD</td><td>28</td><td>199</td><td>0</td><td>0</td><td>0</td><td></td><td>0 0</td><td>0</td><td>0</td></tr><tr><td>Grassmann</td><td>30</td><td>41</td><td>0</td><td>0</td><td>0</td><td></td><td>0 0</td><td>0</td><td>0</td></tr><tr><td>Rotations</td><td>28</td><td>21</td><td>0</td><td>0</td><td>0</td><td></td><td>0 0</td><td>0</td><td>0</td></tr></table>

## D.1.3 Accuracy comparisons

Amortized accuracy In this experiment, we pretrain a flow chart and the two MFMs described in Table 4 on the training set of each geometry, and then evaluate them on the test set. Each is trained once, from one seed, so the error bars measure variation across base points. For the baselines, we evaluate them directly on the test set. We sweep the knot count of the GP and GEORCE, over $K \in \{ 4 , 8 , 1 6 , 3 2 \}$

To estimate the uncertainty in the ratios of Figure 4, we resample the held out base points with replacement 10,000 times, using the same base points for the flow chart and MFM in each sample. We compute the ratio of mean errors for each sample and report the 2.5th and 97.5th percentiles.

Fixed base point accuracy We evaluate the numerical baselines in Figure 3 at the same four base points as the non-amortized flow charts, with indices 26, 44, 67 and 100. For each baseline, we select the knot count with the lowest mean of $\mathcal { R } _ { \mathrm { l o g m a p } }$ over all test base points, which gives K = 16 for the GP and K = 32 for GEORCE on all sixteen geometries. Each flow chart is trained once, from one seed, so the error bars measure variation across base points. The training queries at these base points are separate from the test queries, as described in Section D.3.

Convergence of the baselines Table 6 shows the number of test queries that fail, for the GP and GEORCE at each knot count over all test queries of all test base points.

## D.2 Exponential map experiment details

## D.2.1 Data and geometry

We use the twelve closed form manifolds of Section C.1, three geometries constructed from synthetic datasets using the MFM RBF metric, and the diffusion geometry of alanine dipeptide learned with RMM as described in Section B. After pretraining each RMM geometry, we fix $\varepsilon = 0 . 1$ , so the flow charts, the ODE solvers, and the ground truth all use the cometric $g _ { 0 . 1 , x } ^ { \theta } .$

Synthetic datasets The synthetic datasets have 500,000 points total, and are split into a training, validation and test set with a (70, 10, 20) split. We use the RBF metric from MFM, which we detail in Section C.2. The ground truth is generated in FP64 but the experiments are run in FP32.

Data selection To select endpoints, we sample from each base point based on proximity. For this, we first run clustering on each dataset (except for alanine dipeptide due to its size) and give each point a label corresponding to its connected component. This clustering is done with the HDBSCAN (Campello et al., 2013; McInnes et al., 2017) algorithm from Scikit-learn (Pedregosa et al., 2011). We run it separately on each split, with a minimum cluster size of 1% of the split and 16 samples per core point, and allow a single cluster. Points that are labeled as noise are given the label of their nearest neighbor with a label. We give all points in alanine dipeptide the same cluster label.

Then, given a base point $\mathbf { x } _ { \mathrm { 0 } }$ that is drawn uniformly from the data split, we select an endpoint x by drawing uniformly from the $k = \lfloor \nu \cdot N \rfloor$ Euclidean nearest neighbors with the same cluster label as $\mathbf { x } _ { \mathrm { 0 } }$ . N is the number of points in the split. We use $\nu = 0 . 1$ throughout so that each flow chart sees 10% of the data in the split. The base points and endpoints are drawn from the same data split.

## D.2.2 Flow chart architecture and training details

For the MFM comparison and the flow chart preconditioner, we use the four-layer architecture of Section D.1.1. We train these flow charts for at most 2,000,000 steps, with early stopping on the data geometries. On the data geometries, the learning rate remains constant after warmup, and training stops after 100 validation checks (50,000 steps) without improvement. For the capacity sweep, we use the training procedure of Section D.3, without early stopping. We train with a batch containing 16 base points, with 32 endpoints for each. During evaluation, we measure the average loss over the base points of the validation set, each with a single endpoint drawn from the same procedure with a fixed random seed.

We set ${ \gamma = 0 . 1 \bar { r } / \sqrt { n } } ,$ where r¯ is the median Euclidean distance from a base point to its k-th nearest neighbor within its cluster, which is the radius of the region that endpoints are drawn from. We empirically evaluate this over 64 base points before training.

## D.2.3 Evaluation set generation

This experiment evaluates different methods on triplets $\left( \mathbf { x } _ { 0 } , V _ { x } , \mathbf { x } _ { 1 } \right)$ , where $V _ { x } \approx \log _ { \mathbf { x } _ { 0 } } ( \mathbf { x } _ { 1 } )$ . On the synthetic manifolds, the logmap is known in closed form. On the data geometries, we estimate it using the initial velocity of the GP method of Section D.1.1. On each data geometry, we sample endpoint pairs according to the procedure in Section D.2.1 over the test set. We draw $B = 1 2 8$ base points uniformly from the test set and $M = 5 1 2$ endpoints for each, without replacement, under a fixed random seed, which gives 65,536 triplets per dataset before the discards below. The triplets are computed in FP64 before evaluation and saved to file, except that on the RMM geometries the network $M _ { \varepsilon } ^ { \theta }$ and its derivatives are evaluated in FP32, while the GP iteration and all linear algebra remain in FP64 for numerical stability. We found that in FP32, the GP method consistently diverged. For all of the geometries, we use $K = 5 0$ acceleration nodes, a length scale of $\sqrt { 1 / ( 2 ( K + 1 ) ) }$ , and an acceleration regularizer of $1 0 ^ { - 4 }$

Since the norm of the logmap solution is supposed to equal the geodesic distance between the base point and the query, the logmap solution can be refined by estimating the length and rescaling. Let $\begin{array} { r } { \bar { L } ^ { ( i ) } = \int _ { 0 } ^ { 1 } \| \dot { \mathbf { x } } _ { t } ^ { ( i ) } \| _ { g } } \end{array}$ dt be the length of the GP curve, which we estimate with Simpson’s rule on 101

evenly spaced times. The improved solution is

$$
V _ { x } ^ { ( i ) } = L ^ { ( i ) } \frac { \dot { \bf x } _ { 0 } ^ { ( i ) } } { \| \dot { \bf x } _ { 0 } ^ { ( i ) } \| _ { g } }\tag{51}
$$

Once the curve is fit, we save this refined logmap solution as $V _ { x }$ , together with the entire curve so that we can evaluate against it. We do not include a solution if the GP did not converge, or if (for the synthetic datasets) the GP crosses low data density regions, which we measure with the RBF metric’s network h from Section C.2, keeping a curve only if $h \geq 0 . 5$ at 101 evenly spaced times along it. The GP converges when every squared fixed point residual falls below $1 { \dot { 0 } } ^ { - 4 }$ within 500 iterations. Table 7 lists the number of retained test triplets. On these geometries, the errors measure agreement with the numerical reference, so we compare the GP baseline with the same solver using more acceleration nodes.

Efficient acceleration computation for GP on RMM cometrics For metric parameterized via Section C.4.1, solving for geodesic interpolations for the GP is expensive because every acceleration evaluation differentiates through a cometric solve. However, by taking advantage of the structure of the metric, we can avoid derivative through the inverse. At each of its nodes, the GP evaluates the geodesic acceleration $\ddot { \mathbf { x } } = - \Gamma _ { x } ( \mathbf { x } ) ( \dot { \mathbf { x } } , \dot { \mathbf { x } } )$ from the position x and velocity x˙ of the curve there. If it is computed using the expression from Equation (5), this differentiates the metric $g _ { \varepsilon , x } ^ { \theta } ( \mathbf { x } )$ , which is the inverse of the cometric.

Instead, we compute the acceleration from Hamilton’s equations. Let $\begin{array} { r } { \mathbf { p } = g _ { \varepsilon , x } ^ { \theta } ( \mathbf { x } ) \mathbf { \varepsilon } } \end{array}$ x˙ be the momentum. Under the low rank parameterization, Hamilton’s equations for $\begin{array} { r } { H ( \mathbf { x } , \mathbf { p } ) = \frac { 1 } { 2 } \mathbf { p } ^ { \top } g _ { \varepsilon , x } ^ { \theta } ( \mathbf { x } ) ^ { - 1 } \mathbf { p } } \end{array}$ are

$$
\dot { \mathbf { x } } = \delta \mathbf { p } + ( 1 - \delta ) M _ { \varepsilon } ^ { \theta } ( \mathbf { x } ) ^ { \top } M _ { \varepsilon } ^ { \theta } ( \mathbf { x } ) \mathbf { p } , \qquad \dot { \mathbf { p } } = - \frac { 1 - \delta } { 2 } \nabla _ { \mathbf { x } } \left\| M _ { \varepsilon } ^ { \theta } ( \mathbf { x } ) \mathbf { p } \right\| _ { 2 } ^ { 2 }\tag{52}
$$

where no gradient flows through p. Differentiating x˙ in time gives

$$
\ddot { \mathbf { x } } = ( 1 - \delta ) \left( D _ { \mathbf { \dot { x } } } M _ { \varepsilon } ^ { \theta } ( \mathbf { x } ) ^ { \top } M _ { \varepsilon } ^ { \theta } ( \mathbf { x } ) + M _ { \varepsilon } ^ { \theta } ( \mathbf { x } ) ^ { \top } D _ { \mathbf { \dot { x } } } M _ { \varepsilon } ^ { \theta } ( \mathbf { x } ) \right) \mathbf { p } + g _ { \varepsilon , x } ^ { \theta } ( \mathbf { x } ) ^ { - 1 } \dot { \mathbf { p } }\tag{53}
$$

where $D _ { \dot { \mathbf { x } } } M _ { \varepsilon } ^ { \theta } ( \mathbf { x } )$ is the directional derivative of $M _ { \varepsilon } ^ { \theta } ( \mathbf { x } )$ at $\mathbf { x }$ along ${ \dot { \mathbf { x } } } .$ . This expression of the acceleration is cheap to compute because it requires a single linear solve, for p, which is never differentiated. Every derivative can be computed cheaply with Jacobian vector products of vector Jacobian products.

## D.2.4 Methods

The baselines simulate the geodesic with initial position $\mathbf { x } _ { \mathrm { 0 } }$ and velocity $V _ { x }$ by integrating Hamilton’s equations for Hamiltonian $\mathbf { \bar { \chi } } _ { H ( \mathbf { x } , \mathbf { p } ) } = \textstyle { \frac { 1 } { 2 } } \mathbf { p } ^ { \top } g _ { x } ( \mathbf { \bar { x } } ) ^ { - 1 } \mathbf { p }$

$$
\dot { \mathbf { x } } _ { t } = g _ { x } ( \mathbf { x } _ { t } ) ^ { - 1 } \mathbf { p } _ { t } , \qquad \dot { \mathbf { p } } _ { t } = - \frac { 1 } { 2 } \nabla _ { \mathbf { x } } \left( \mathbf { p } _ { t } ^ { \top } g _ { x } ( \mathbf { x } _ { t } ) ^ { - 1 } \mathbf { p } _ { t } \right) , \qquad \mathbf { p } _ { 0 } = g _ { x } ( \mathbf { x } _ { 0 } ) V _ { x } \mathbf { p } _ { t } ,\tag{54}
$$

from $t = 0 \mathrm { t o } t = 1$ , so that $\mathbf { x } _ { 1 } = \exp _ { \mathbf { x } _ { 0 } } ( V _ { x } )$ . On the RMM geometry, we directly parameterize the cometric, so each update does not require a matrix inversion. Only the initial momentum $\mathbf { p } _ { 0 }$ requires an inversion. We integrate with four fixed-step solvers from diffrax (Kidger, 2021) of increasing order, Euler, Heun, Tsit5 (Tsitouras, 2011), and Dopri8 (Prince and Dormand, 1981), which cost one, two, six, and thirteen evaluations of Equation (54) per step, plus one more at the first step for Tsit5 and Dopri8 and one evaluation of the cometric at $\mathbf { x } _ { \mathrm { 0 } }$ to form p<sub>0</sub>. We run each solver with S equal steps over [0, 1]. For a solver that costs c evaluations per step, we sweep 5 step counts spaced geometrically from 1 to $\lfloor 1 2 8 / c \rfloor$ and rounded, which gives $S \in \bar { \{ 1 , 3 , 1 1 , 3 8 , \bar { 1 } 2 8 \} }$ for Euler, {1, 3, 8, 23, 64} for Heun, {1, 2, 5, 10, 21} for Tsit5, and {1, 2, 3, 5, 9} for Dopri8. These choices are so that no baseline evaluates the cometric more than 129 times. On the data geometries, we also use the solver’s interpolation to record the state at the 50 times used for the reference curve. A query fails if the solver returns a nonfinite endpoint. The solvers are run in FP32.

The flow chart computes $\exp _ { \mathbf { x } _ { 0 } } ( V _ { x } ) = f _ { \mathbf { x } _ { 0 } } ^ { \theta } \left( D h _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { x } _ { 0 } ) V _ { x } \right)$ as described in Section 3.

## D.2.5 Evaluation metrics

For each method, we compute $\mathcal { E } _ { \mathrm { e x p m a p } } ( \mathbf { x } _ { 0 } )$ and $\mathcal { R } _ { \mathrm { e x p m a p } } ( \mathbf { x } _ { 0 } )$ as defined in Section 5.1. The numerator uses the queries in $\mathcal { T } ( \mathbf { x } _ { \mathrm { 0 } } )$ for which the method returns a finite endpoint. The reference scale $r _ { \mathrm { c h o r d } } ( \mathbf { x } _ { 0 } )$ uses all queries in $\mathcal { I } ( \mathbf { x } _ { 0 } )$ and is shared across methods. We average $\mathcal { R } _ { \mathrm { e x p m a p } } ( \mathbf { x } _ { 0 } )$ over base points with valid readings and report its mean and standard error multiplied by 100.

We time every method with the procedure of Section D. Let $\tau ( \mathbf { x } _ { 0 } )$ be the wall clock time to compute the endpoints at a test base point $\mathbf { x } _ { 0 } .$ , divided by the number of queries, so that $\tau ( \mathbf { x } _ { 0 } )$ is a time per query. Before discarding queries without a valid reference solution, there are 1,024 queries per base point on the closed form manifolds and 512 on the data geometries. For the flow chart, $\tau ( \mathbf { x } _ { 0 } )$ includes encoding $\mathbf { x } _ { \mathrm { 0 } }$ once, and the chart reuses this encoding for all of its queries. For the ODE solvers, $\tau ( \mathbf { x } _ { 0 } )$ includes forming the initial momenta. Every method evaluates the queries of a base point in parallel on a GPU, and we report $\tau ,$ the mean of $\tau ( \mathbf { x } _ { 0 } )$ over test base points.

## D.3 Capacity and fixed base point experiments

We vary the capacity of the flow chart and compare amortized and non-amortized training on the closed form manifolds, the three synthetic datasets, and alanine dipeptide. We use both the low and high dimensional closed form manifolds, where the logarithmic and exponential maps are known in closed form. On the data geometries, we use the numerical references of Section D.2. See Section C.1, Section C.2 and Section B for a description of each dataset, the geometry they define, and how the training, validation and test sets are generated. Each dataset produces a set of B base points, and for each base point, produces N query points from which to train or evaluate with. An element of each dataset consists of a triplet of the form $( \mathbf { x } _ { 0 } , V _ { x } , \mathbf { x } )$ , where $V _ { x } = \log _ { \mathbf { x } _ { 0 } } ( \mathbf { x } )$ and $\mathbf { x } = \exp _ { \mathbf { x } _ { 0 } } ( V _ { x } )$ are the solution to the logarithmic and exponential maps respectively.

We train flow charts on this data in two different settings. The first is an amortized setting over base points, where we train on $B _ { \mathrm { t r a i n } }$ base points for each dataset, each with $N _ { \mathrm { t r a i n } }$ associated query points. During evaluation, we compute $\mathcal { R } _ { \mathrm { l o g m a p } }$ and $\mathcal { R } _ { \mathrm { e x p m a p } }$ over a set of $B _ { \mathrm { t e s t } }$ held out test base points, each with $N _ { \mathrm { t e s t } }$ unseen query points before discarding those without a valid reference solution.

The second is a non-amortized setting where we train a flow chart at a single test base point. These $B _ { \mathrm { s i n g l e } } = 4$ test base points are selected at random (with a fixed random seed) from the $B _ { \mathrm { t e s t } }$ held out test base points from each dataset. For each of these base points, we construct a training set of $N _ { \mathrm { s i n g l e } }$ query points that is separate from its $N _ { \mathrm { t e s t } }$ test query points. On the closed form datasets, we sample $N _ { \mathrm { s i n g l e } }$ new query points at the base point with the sampling procedure of Section C.1, independently of the test query points. On the synthetic and bio datasets, the training query points are the $N _ { \mathrm { s i n g l e } }$ nearest neighbors of the base point in the training set, selected as in Section D.2.1. The base point is a test point and its training query points are training points, so a flow chart never trains on a test query point. Table 7 lists $N _ { \mathrm { s i n g l e } }$ for each dataset.

For Figure 3, we compute $\mathcal { R } _ { \mathrm { l o g m a p } }$ at each of these four base points using the same test queries as for the amortized flow chart.

## D.3.1 Flow-chart architecture

In all of these experiments, we use a modified version of the conditional neural spline flow of Section D.1.1. We keep the conditioned elementwise scale and shift and the invertible linear map that are placed before each coupling layer, and change the coupling layer and the width of its conditioner. Each coupling layer transforms an input v into $s \cdot \mathrm { s p l i n e } ( v ) + t .$ , where the spline is a rational quadratic spline with 8 bins on [−10, 10], and the 23 spline parameters, the scale s and the shift t are all predicted by a FiLM conditioned residual network with 3 blocks and hidden width of 64. The scale s is passed through a softplus so that the flow chart is close to the identity transformation at initialization.

In the amortized setting, the base point encoder is a residual network with 8 gated residual block of width 64 that encodes the base point into a 64-dimensional feature vector for all coupling layers. In the non-amortized setting, the base point is fixed, so the base point encoder has a single gated residual block. We vary the capacity of the amortized flow chart through the number of coupling layers, $L \in \{ 4 , 8 , 1 6 , 3 2 \}$ , and use $L = 3 2$ for the accuracy comparison at fixed base points.

Table 7: Sizes of the training and test sets. On the synthetic and bio datasets, every point of the training set is a base point and its query points are its $ { \lfloor 0 . 1 B _ { \mathrm { t r a i n } } } _ { . }$ ⌋ nearest neighbors with the same cluster label. The last column is the number of test triplets that remain out of $B _ { \mathrm { t e s t } } N _ { \mathrm { t e s t } }$ after the discards of Section D.2.
<table><tr><td>Dataset</td><td>n</td><td> $B _ { \mathrm { t r a i n } }$ </td><td> $N _ { \mathrm { t r a i n } }$ </td><td> $B _ { \mathrm { t e s t } }$ </td><td> $N _ { \mathrm { t e s t } }$ </td><td> $N _ { \mathrm { s i n g l e } }$ </td><td>Test triplets</td></tr><tr><td>Sphere</td><td>2/30</td><td>2,048</td><td>1,024</td><td>128</td><td>1,024</td><td>262,144</td><td>131,072</td></tr><tr><td>Flat torus</td><td>2/30</td><td>2,048</td><td>1,024</td><td>128</td><td>1,024</td><td>262,144</td><td>131,072</td></tr><tr><td>Hyperbolic</td><td>2/30</td><td>2,048</td><td>1,024</td><td>128</td><td>1,024</td><td>262,144</td><td>131,072</td></tr><tr><td>SPD</td><td>3/28</td><td>2,048</td><td>1,024</td><td>128</td><td>1,024</td><td>262,144</td><td>131,072</td></tr><tr><td>Grassmann</td><td>4/30</td><td>2,048</td><td>1,024</td><td>128</td><td>1,024</td><td>262,144</td><td>131,072</td></tr><tr><td>Rotations</td><td>3/28</td><td>2,048</td><td>1,024</td><td>128</td><td>1,024</td><td>262,144</td><td>131,072</td></tr><tr><td>Disk</td><td>2</td><td>350,000</td><td>35,000</td><td>128</td><td>512</td><td>35,000</td><td>65,536</td></tr><tr><td>Two moons</td><td>2</td><td>350,000</td><td>35,000</td><td>128</td><td>512</td><td>35,000</td><td>62,221</td></tr><tr><td>Nested circles</td><td>2</td><td>350,000</td><td>35,000</td><td>128</td><td>512</td><td>35,000</td><td>65,041</td></tr><tr><td>Alanine dipeptide</td><td>30</td><td>525,000</td><td>52,500</td><td>128</td><td>512</td><td>52,500</td><td>65,529</td></tr></table>

## D.3.2 Training

We train every flow chart with the procedure of Section D.1.1, with a budget of 2,000,000 steps, learning rate decay and no early stopping. All non-amortized flow charts reported here and all amortized flow charts with $L = 3 2$ reached the full budget. On two moons, the amortized fits with $L = 4$ and $L = 8$ diverged after 1,154,500 and 1,292,500 steps respectively. We evaluate these flow charts using their best checkpoints selected before divergence. On the geometries whose cometric is learned with RMM, we fix $\varepsilon = 0 . 1$ . In both settings, we set $\gamma = 0 . 1 r _ { \mathrm { m a x } } / \sqrt { n }$ on the closed form datasets as in Section D.1.1, and $\gamma = 0 . 1 \bar { r } / \sqrt { n }$ on the synthetic and bio datasets as in Section D.2. In the amortized setting, each training batch contains 16 base points with 32 query points each. In the non-amortized setting there is a single base point, so each training batch contains 512 query points drawn uniformly from the $N _ { \mathrm { s i n g l e } }$ training points of that base point. On the closed form datasets, the validation loss in the non-amortized setting is evaluated on a validation set containing 1,024 independently sampled points. On the synthetic and bio datasets, this validation set consists of the $ { \lfloor 0 . 1 \bar { N } _ { \mathrm { v a l } _ { - } } }$ nearest neighbors of the base point in the validation set, selected as in Section D.2.1, where $N _ { \mathrm { v a l } }$ is the number of points in the validation set.

## D.3.3 Evaluation

We evaluate every flow chart in the same way as the flow charts of the experiments above. On every geometry, we compute the mean of $\mathcal { R } _ { \mathrm { l o g m a p } }$ or $\mathcal { R } _ { \mathrm { e x p m a p } }$ across base points as described in Sections D.1.2 and D.2. On the synthetic and bio datasets, the ground truth normal coordinates of a test triplet are $\mathbf { s } = g _ { x } ( \mathbf { x } _ { 0 } ) ^ { \frac { 1 } { 2 } } V _ { x }$

In the amortized setting, we evaluate the flow chart at all $B _ { \mathrm { t e s t } }$ test base points. In the non-amortized setting, we evaluate each flow chart at its own base point and report the mean over the $B _ { \mathrm { s i n g l e } }$ flow charts. For the accuracy comparison at fixed base points, we evaluate the flow charts and numerical baselines at the same $B _ { \mathrm { s i n g l e } }$ base points.

## D.3.4 Baselines

Closed form manifolds ODE Solvers We run the ODE solvers of Section D.2 on the closed form manifolds in both low and high dimensions. All $N _ { \mathrm { t e s t } }$ queries of a test base point are solved in a single batch.

Closed form manifolds MFM and GP, GEORCE See Section D.1.

Data manifolds ODE Solvers See Section D.2.

Table 8: Sizes of the MFM baselines on the synthetic and bio datasets.
<table><tr><td></td><td></td><td colspan="2">Flow chart</td><td colspan="3">MFM, parameter matched</td><td colspan="3">MFM, time matched</td></tr><tr><td>Dataset</td><td>n</td><td>Parameters</td><td>µs</td><td>Size</td><td>Parameters</td><td>µs</td><td></td><td>Size Parameters</td><td>µs</td></tr><tr><td>Disk</td><td>2</td><td>187,504</td><td>1.85</td><td> $6 4 \times 1 5$ </td><td>188,674</td><td>2.20</td><td> $6 4 \times 1 2$ </td><td>151,042</td><td>1.82</td></tr><tr><td>Two moons</td><td>2</td><td>187,504</td><td>1.92</td><td> $6 4 \times 1 5$ </td><td>188,674</td><td>2.23</td><td> $6 4 \times 1 2$ </td><td>151,042</td><td>1.87</td></tr><tr><td>Nested circles</td><td>2</td><td>187,504</td><td>1.89</td><td> $6 4 \times 1 5$ </td><td>188,674</td><td>2.22</td><td> $6 4 \times 1 2$ </td><td>151,042</td><td>1.87</td></tr><tr><td>Alanine dipeptide 30</td><td></td><td>305,008</td><td>2.34</td><td> $1 2 8 \times 6$ </td><td>309,790</td><td>2.74</td><td> $6 4 \times 7$ </td><td>93,726</td><td>2.29</td></tr></table>

Table 9: Parameter counts of the amortized flow charts used in the capacity sweep for each number of coupling layers L.
<table><tr><td>Datasets</td><td>n</td><td> $L = 4$ </td><td>8</td><td>16</td></tr><tr><td>Sphere, torus, hyperbolic</td><td>2</td><td>319,232</td><td>533,760</td><td>962,816 1,820,928</td></tr><tr><td>SPD, rotations</td><td>3</td><td>326,616</td><td>548,464</td><td>992,160 1,879,552</td></tr><tr><td>Grassmann</td><td>4</td><td>334,016</td><td>563,200 1,021,568</td><td>1,938,304</td></tr><tr><td>SPD, rotations</td><td>28</td><td>516,416</td><td>926,464 1,746,560</td><td>3,386,752</td></tr><tr><td>Sphere, torus, hyperbolic, Grassmann</td><td>30</td><td>532,032</td><td>957,568 1,808,640</td><td>3,510,784</td></tr><tr><td>Disk, two moons, nested circles</td><td>2</td><td>319,232</td><td>533,760 962,816</td><td>1,820,928</td></tr><tr><td>Alanine dipeptide</td><td>30</td><td>532,032</td><td>957,568 1,808,640</td><td>3,510,784</td></tr></table>

Data manifolds MFM and GP, GEORCE We run the methods of Section D.1.1 on the synthetic and bio datasets, and evaluate them as in Section D.1.2 on the test triplets of Section D.2 that were not discarded. The batch sizes of the GP and GEORCE are selected from {64, 128, 256, 512}.

The two MFMs are matched to the flow chart of Section D.2 on each dataset, with the time measured on the 512 queries of a test base point, and Table 8 lists their sizes. They are trained on the same training pairs as that flow chart, with the same training procedure as Section D.1. We train these MFMs for 2,000,000 steps with learning rate decay and no early stopping.

## D.3.5 Experiments

This experiment measures how the errors of a flow chart depend on its capacity, and how much of the error is due to amortization. For every dataset of Table 7, we use an amortized flow chart at each $L \in \{ 4 , 8 , 1 6 , 3 2 \}$ for the runtime comparisons. For the accuracy comparison at fixed base points, we use $B _ { \mathrm { s i n g l e } } = 4$ non-amortized flow charts with $L = 3 2 .$ , one at each base point. Every other choice is held fixed across L, and a flow chart in the non-amortized setting differs from the amortized flow chart of the same L in its base point encoder and its training data alone. Each flow chart is trained once, from one seed.

We compare against the baselines of the experiments above, evaluated on the same test triplets. For the logarithmic map, these are the GP, GEORCE and the two MFMs of Section D.1.1, and for the exponential map they are the ODE solvers of Section D.2. The two MFMs are selected to have the same parameter count and time as the flow chart used in the original experiment, and not the size of the ablation models.

Table 9 contains the parameter count of the amortized flow chart for each dimension and L.

## D.3.6 Scaling with dimension

Figure 7 shows that the cost of a pretrained flow chart also scales favorably with dimension compared with the numerical baselines. Across the closed form manifolds that have low and high dimensional variants, the smallest and largest chart sizes only take 1.30 to 1.63 times longer to run in high dimensions while the GP grows by 3.0 to 52× and GEORCE by 3.8 to $2 7 \times$ (at $\check { K } = 1 6 )$

![](images/ab80f0561eb6c05d338896b3e4f168ad2b64644c4b50cde936ddcdff9a365974.jpg)  
Figure 7: Increase in time per logmap query from the low dimensional to the high dimensional version of each manifold. Each method uses the same architecture or knot count in both dimensions.

## D.4 Matching and solver resolution

To compare runtimes at matched accuracy, we set the error threshold $\mathcal { R } _ { \star }$ to the larger of the minimum mean errors in the two sweeps. The mean error is the mean of $\mathcal { R } _ { \mathrm { l o g m a p } }$ or $\mathcal { R } _ { \mathrm { e x p m a p } }$ across base points for the logarithmic or exponential map respectively. Within each sweep, we select the fastest setting whose mean error is at most R<sub>⋆</sub>. We report the speedup as the runtime of the baseline divided by that of the flow chart. This comparison uses the test errors to select settings and describes their measured performance. We do not interpolate between settings. To compare errors at matched cost, we fix the $\bar { L } = 3 2$ flow chart and select the ODE setting with the lowest error among those whose runtime does not exceed that of the chart. Table 10 lists the settings selected for the comparisons in Figure 5.

On the two dimensional sphere, several ODE settings have very large finite errors, and the errors do not consistently decrease as the number of steps increases. The matched comparison in Figure 5 uses Heun with 64 steps. These results show instability on this geometry, but we have not identified its cause. The conclusions about the exponential map in high dimensions do not depend on this geometry.

Table 10: Settings selected for the runtime comparisons in Figure 5. For each numerical method, L is the number of coupling layers in the chart used in that comparison. K is the number of knots for GP or GEORCE, and S is the number of ODE steps. Each comparison selects the fastest chart and numerical settings whose mean normalized RMSE is at most $\mathcal { R } _ { \star } ^ { \bar { } }$
<table><tr><td></td><td colspan="4">Logarithmic map</td><td colspan="3">Exponential map</td></tr><tr><td></td><td colspan="2">GP</td><td colspan="2">GEORCE</td><td colspan="3">ODE</td></tr><tr><td>Geometry</td><td>L</td><td>K</td><td>L</td><td>K</td><td>L</td><td>Solver</td><td>S</td></tr><tr><td>Sphere 2</td><td>16</td><td>16</td><td>4</td><td>32</td><td>16</td><td>Heun</td><td>64</td></tr><tr><td>Flat torus 2</td><td>8</td><td>16</td><td>4</td><td>32</td><td>16</td><td>Tsit5</td><td>2</td></tr><tr><td>Hyperbolic 2</td><td>8</td><td>16</td><td>4</td><td>32</td><td>16</td><td>Tsit5</td><td>2</td></tr><tr><td>SPD 3</td><td>16</td><td>16</td><td>4</td><td>32</td><td>32</td><td>Dopri8</td><td>1</td></tr><tr><td>Grassmann 4</td><td>32</td><td>16</td><td>4</td><td>32</td><td>32</td><td>Dopri8</td><td>1</td></tr><tr><td>Rotations 3</td><td>32</td><td>16</td><td>4</td><td>32</td><td>32</td><td>Tsit5</td><td>2</td></tr><tr><td>Sphere 30</td><td>32</td><td>8</td><td>32</td><td>32</td><td>32</td><td>Heun</td><td>8</td></tr><tr><td>Flat torus 30</td><td>32</td><td>16</td><td>16</td><td>32</td><td>32</td><td>Dopri8</td><td>2</td></tr><tr><td>Hyperbolic 30</td><td>32</td><td>8</td><td>32</td><td>32</td><td>32</td><td>Tsit5</td><td>2</td></tr><tr><td>SPD 28</td><td>32</td><td>8</td><td>32</td><td>32</td><td>32</td><td>Heun</td><td>8</td></tr><tr><td>Grassmann 30</td><td>32</td><td>8</td><td>32</td><td>32</td><td>32</td><td>Dopri8</td><td>1</td></tr><tr><td>Rotations 28</td><td>32</td><td>8</td><td>32</td><td>16</td><td>32</td><td>Tsit5</td><td>2</td></tr><tr><td>Disk</td><td>8</td><td>16</td><td>4</td><td>32</td><td>4</td><td>Dopri8</td><td>1</td></tr><tr><td>Two moons</td><td>16</td><td>16</td><td>4</td><td>32</td><td>16</td><td>Tsit5</td><td>2</td></tr><tr><td>Nested circles</td><td>4</td><td>16</td><td>4</td><td>32</td><td>16</td><td>Dopri8</td><td>2</td></tr><tr><td>Alanine dipeptide</td><td>32</td><td>8</td><td>32</td><td>32</td><td>32</td><td>Heun</td><td>3</td></tr></table>

## D.4.1 Exponential map errors and runtime budget

For the exponential map, the flow chart with $L = 3 2$ achieves a mean $1 0 0 \times \mathcal { R } _ { \mathrm { e x p m a p } }$ between 0.15 and 2.51 across the seven geometries with $n \geq 2 8$ , including $1 . 9 9 \pm 0 . 2 9$ on alanine dipeptide. We also compare flow charts and ODE solvers when the ODE solvers are limited to the runtime of the $L = 3 2$ chart. Within this budget, the mean $\mathcal { R } _ { \mathrm { e x p m a p } }$ of the most accurate solver is 6.1–195 times that of the flow chart on the high dimensional closed form manifolds, and 3.5 times that of the flow chart on alanine dipeptide. The solvers can reach lower errors if given more time. Figure 9 gives the errors and runtimes at every chart size and ODE step count.

## D.5 Flow chart preconditioner

For Table 1, we use the pretrained flow charts with four coupling layers described in Section D.1.1. For each base point, we express the metric in the chart coordinates using Equation (1), encode each query, and run GEORCE between the origin and the encoded query. We map the initial velocity back to data coordinates through $D f _ { \mathbf { x } _ { 0 } } ^ { \theta } ( \mathbf { 0 } )$ to obtain the logmap estimate. We use the same knot counts and stopping criterion as GEORCE in data coordinates. We report the mean of $\mathcal { R } _ { \mathrm { l o g m a p } }$ across base points and the ratio of the mean errors at each knot count. GEORCE successfully returns a solution for every query in this comparison in both data and chart coordinates.

At $K = 3 2 .$ the mean of $\mathcal { R } _ { \mathrm { l o g m a p } }$ is reduced by $2 . 3 \mathrm { - 3 . 6 \times }$ on the six high dimensional closed form manifolds, and by $2 . 2 \mathrm { - } 5 . 9 \times$ on the data geometries. The improvement holds at all four knot counts, with factors of 2.2–8.1 across the 40 settings. This suggests that the flow chart learns a space in which the numerical errors associated with GEORCE are reduced. We compare the methods at equal knot counts, so these evaluations do not establish a runtime improvement.

## E Complete experimental results

We summarize the results over all geometries and tested settings in three figures. Figures 8 and 9 show the accuracy and time per query for the logarithmic and exponential maps respectively. Figure 10 shows the effect of flow chart capacity and amortization.

![](images/185abf7b206cbe4baa47fe984d7f98caf3c7f6916fcb52659517abdfac1f848c.jpg)  
Figure 8: Logarithmic map accuracy and runtime across all sixteen geometries. Each point gives the mean of $1 0 0 \times \mathcal { R } _ { \mathrm { l o g m a p } }$ , with one standard error across base points. The points for the four flow chart sizes and the four knot counts of the GP and GEORCE are joined by lines. The open diamond denotes the smaller flow chart used for the MFM comparison, shown alongside both MFM baselines. Both axes use logarithmic scales, with ranges chosen separately for each geometry. A black cross marks a setting with failed queries, which are excluded when computing its error.

![](images/0e6b66399ee235ded3be63718524889b1f78a6fdb0d80ad6b2382b9468e34d62.jpg)  
Figure 9: Exponential map accuracy and runtime across all sixteen geometries. Each point gives the mean of $1 0 0 \times \mathcal { R } _ { \mathrm { e x p m a p } } ,$ , with one standard error across base points. The curves show every flow chart size and ODE step count listed in the legend. Both axes use logarithmic scales, with ranges chosen separately for each geometry. A black cross marks a setting with nonfinite endpoints, which are excluded when computing its error. We retain all finite errors. The large errors on the two dimensional sphere show that several ODE settings are unstable on this geometry. All flow chart queries with valid references return finite endpoints.

![](images/d14541c223df813bb6bc093b0228d1aded55680e1a123aa45228f861be425a76.jpg)  
Figure 10: Effect of flow chart capacity and amortization. The panels show the mean of $1 0 0 \times \mathcal { R } _ { \mathrm { l o g m a p } }$ for amortized and non-amortized flow charts, evaluated on the same four test base points at each chart size. The two panels share a logarithmic color scale.