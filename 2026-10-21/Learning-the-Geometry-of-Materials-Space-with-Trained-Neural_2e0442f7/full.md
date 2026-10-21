# Learning the Geometry of Materials Space with Trained Neural Tangent Kernels

Daniel Kaplan,<sup>1,</sup> <sup>2,</sup> <sup>3,</sup> <sup>4,</sup> <sup>∗</sup> Gabriel Kotliar,<sup>1,</sup> <sup>2</sup> and Anirvan M Sengupta<sup>5,</sup> <sup>6,</sup> <sup>2</sup>

<sup>1</sup>Center for Materials Theory, Department of Physics and Astronomy,

Rutgers University, Piscataway 08854 New Jersey USA

<sup>2</sup>Department of Physics and Astronomy, Rutgers University, Piscataway 08854 New Jersey USA

<sup>3</sup>Department of Physics, University of Central Florida, Orlando, Florida 32816, USA

<sup>4</sup>NanoScience and Technology Center, University of Central Florida, Orlando, Florida 32816, USA

<sup>5</sup>Center for Computational Quantum Physics, Flatiron Institute, New York, NY 10010

Neural networks trained to predict materials properties learn representations whose physical content can be dificult to extract. We use the neural tangent kernel (NTK), evaluated at trained parameters, to recover the geometry learned by a network that predicts superconducting tran sition temperatures from elemental composition alone. Kernel ridge regression with the trained NTK closely reproduces the network’s predictive performance and substantially outperforms regression with the initial kernel. Spectral analysis and multidimensional scaling reveal a geometry dominated by a small number of directions. Clustering and local neighborhood analysis organize compounds into recognizable superconducting families and connect materials with related chemical and correlated-electron properties despite substantial diferences in $T _ { c } .$ . These results establish trained tangent kernels as practical probes of learned materials representations, connecting property prediction with the exploration of materials space.

## I. INTRODUCTION

Neural networks trained on materials datasets can learn representations that support accurate predictions for unseen compounds [1, 2]. The reuse of learned features across related tasks underlies transfer learning, which has become an important tool in materials science [3–5]. Understanding the physical content of these representations remains a central challenge in scientific machine learning [6]. Learning meaningful representations is a long-standing goal in machine learning. Convolutional networks learn hierarchical features [7, 8], while independent component analysis and sparse coding identify statistical structure directly from data [9, 10]. These approaches helped establish modern representation learning [11]. In materials science, selfsupervised learning, manifold learning, and generative approaches increasingly exploit structural, graph-based, or multimodal information to organize and explore materials space [12–15].

Our earlier work [16] demonstrated that competitive prediction of superconducting transition temperatures is possible using elemental composition alone, without engineered descriptors such as Magpie [17, 18] or the crystal structures used by many graph-based materials representations [19, 20]. The success of this predictor raises a complementary question: what physical relationships between materials has the network acquired through supervised training, and how can they be recovered? Here we extract and interpret this learned representation from the trained predictor itself, without introducing additional descriptors or auxiliary representation-learning objectives.

We use the neural tangent kernel (NTK) evaluated at the trained parameters as a probe of this representation. Originally introduced to analyze optimization in wide neural networks [21], the NTK can evolve substantially during finite-width training and become aligned with the learned task [22–24]. Its geometry therefore provides access to relationships acquired during training, extending its use from the analysis of optimization to the interpretation of learned materials representations.

For a scalar predictor f(x, θ) with trained parameters θ<sup>∗</sup>, the tangent feature map $\phi ( x ) = \nabla _ { \theta } f ( x , \theta ) | _ { \theta = \theta ^ { * } }$ defines the kernel $K ( x , x ^ { \prime } ) = \phi ( x ) ^ { \top } \phi ( x ^ { \prime } )$ The corresponding distance, $d ( x , x ^ { \prime } ) = \| \phi ( x ) - \phi ( x ^ { \prime } ) \| _ { 2 }$ , compares the local responses of the predictions to parameter perturbations. Compounds are close in this geometry when their predictions respond similarly to small changes in the trained parameters, even when their compositional vectors have little or no elemental overlap.

We show that kernel ridge regression with the trained NTK achieves predictive performance comparable to that of the original network, while regression with the initial kernel performs substantially worse. Spectral analysis and multidimensional scaling reveal a geometry dominated by a small number of directions. Applying unsupervised clustering to this embedding identifies recognizable regions of materials space, including cuprates, iron-based superconductors, and non-superconducting compounds. The cluster names are assigned after clustering, allowing us to interpret the physical organization encoded in the learned geometry.

Local neighborhoods reveal further relationships between chemically distinct materials. A particularly striking example is the proximity of CeCoIn<sub>5</sub> and ${ \mathrm { Y b R h } } _ { 2 } { \mathrm { S i } } _ { 2 } \colon$ these compounds share no constituent elements, yet both exhibit heavy-fermion behavior and are central to the study of correlated electron physics [25 27]. Their connection emerges without supplying this phenomenology as a training label.

![](images/f5f2e7b38ba60fb7be9cfe701d82a172bda2a557cf2e578e0c2def8d91fb116a.jpg)  
FIG. 1 : Schematic overview of the workflow. Minimal compositional descriptors are used to train a neural network predictor for superconducting properties. The neural tangent kernel (NTK) evaluated at trained parameters induces a task-adaptive similarity geometry on materials space. Dimensional reduction and clustering of this geometry reveal semantically meaningful structure learned through supervised training.

These neighborhoods extend the use of the predictor to identifying related compounds and locating regions in which nearby materials span a broad range of $T _ { c }$ . Such regions provide targets for further materials exploration. The trained NTK thus turns an existing property predictor into a tool for navigating materials space and provides a learned geometry for subsequent transfer learning and materials discovery. An overview of the workflow is shown in Fig. 1.

## II. FROM INFLUENCE FUNCTIONS TO TRAINED TANGENT KERNELS

A natural data-space definition of interpretability is influence: how would the prediction at one material change if another material in the training set were infinitesimally reweighted. This question is especially useful in materials science, where interpretation often proceeds by identifying representative compounds, nearby families, and physically analogous systems.

Let $z _ { i } = ( x _ { i } , y _ { i } )$ denote a training example, and $y _ { i }$ the empirical ground truth. For the purposes of what follows, $x _ { i }$ will be a vector of dimension $x \in \mathbb { R } ^ { 1 2 0 } [ 2 8 ]$ and $y _ { i }$ is a real number denoting $T _ { c }$ . For the empirical loss,

$$
\mathcal { L } ( \theta ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell ( f _ { \theta } ( x _ { i } ) , y _ { i } ) ,\tag{1}
$$

we arrive at $\theta ^ { * }$ as the trained parameters. Following the influence-function construction [29], we up-weight point

i by an infinitesimal amount $\epsilon ,$

$$
\theta _ { \epsilon , i } = \arg \operatorname* { m i n } _ { \theta } \left[ \mathcal { L } ( \theta ) + \epsilon \ell ( f _ { \theta } ( x _ { i } ) , y _ { i } ) \right] .\tag{2}
$$

To first order in $\epsilon ,$

$$
\left. \frac { d \theta _ { \epsilon , i } } { d \epsilon } \right| _ { \epsilon = 0 } = - H _ { \theta ^ { * } } ^ { - 1 } \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } ) ,\tag{3}
$$

where $H _ { \theta ^ { * } } = \nabla _ { \theta } ^ { 2 } \mathcal { L } ( \theta ^ { * } )$ . The corresponding change in the prediction at another point $x _ { j } \ [ 2 9 ]$ is

$$
\begin{array} { r } { \mathbb { Z } _ { i \to j } = - \nabla _ { \theta } f _ { \theta ^ { * } } ( x _ { j } ) ^ { T } H _ { \theta ^ { * } } ^ { - 1 } \nabla _ { \theta } \ell _ { i } ( \theta ^ { * } ) . } \end{array}\tag{4}
$$

For squared loss, with residual $r _ { i } = f _ { \theta ^ { * } } ( x _ { i } ) - y _ { i }$ , this becomes

$$
\mathcal { T } _ { i  j } = - r _ { i } \phi _ { j } ^ { T } H _ { \theta ^ { * } } ^ { - 1 } \phi _ { i } , \qquad \phi _ { i } = \nabla _ { \theta } f _ { \theta ^ { * } } ( x _ { i } ) .\tag{5}
$$

Thus the influence function decomposes into a residual factor and a Hessian measure for similarity between two data points. In our case, these data points are in fact diferent materials.

Often, as for our $f ( x , \theta ^ { * } )$ , computing this expression exactly is prohibitive: the Hessian is large, potentially ill-conditioned, and expensive to invert. We are therefore motivated to replace the full nonlinear model by its local linearization at the trained parameters,

$$
\begin{array} { r } { f _ { \theta ^ { * } + \delta \theta } ( x ) \simeq f _ { \theta ^ { * } } ( x ) + \phi ( x ) ^ { T } \delta \theta . } \end{array}\tag{6}
$$

Given the appearance of $\phi ( x ) ^ { T }$ , it is useful to define the Gram object, $K _ { i j } \phi _ { i } ^ { T } \phi _ { j }$ , which constitutes a kernel in feature space. The connection to influence becomes explicit under the Gauss-Newton approximation $H _ { \theta ^ { * } } \simeq$ $J ^ { \bar { T } } J + \lambda I .$ , where J is the Jacobian with rows $\phi _ { i } ^ { T }$ . In this approximation, considering the Sherman-Morrison-Woodbury identity [30],

$$
J ( J ^ { T } J + \lambda I ) ^ { - 1 } J ^ { T } = K ^ { * } ( K ^ { * } + \lambda I ) ^ { - 1 } .\tag{7}
$$

Consequently, the regularized influence of training point i on prediction $j$ is controlled by

$$
\begin{array} { r } { \mathcal { T } _ { i  j } ^ { \lambda } \simeq - r _ { i } [ K ( K + \lambda I ) ^ { - 1 } ] _ { j i } . } \end{array}\tag{8}
$$

Thus the trained NTK, being a positive-semidefinite kernel, is an object whose spectral geometry determines the regularized influence operator of the linearized network. In the analysis below, we use $K$ and distances derived from it as a stable, label-independent proxy for this influence geometry.

## III. THE TRAINED NEURAL TANGENT KERNEL AND REPRESENTATION FIDELITY

The trained NTK can be shown to serve functions which are broadly related to the tasks and modalities of the neural network itself. For example, we first test the ability of the NTK for prediction. We carry out the following procedure and compare its performance with that of the original network. Let our inputs be x and corresponding labels be (here, $T _ { c } )$

The predictive mean for a new test point $x _ { * }$ in this procedure would be given by:

$$
f ( x _ { * } ) = K ( x _ { * } , X ) \left( K ( X , X ) + \lambda I \right) ^ { - 1 } y\tag{9}
$$

where $K ( X , X )$ is the $N \times N$ post-training Gram matrix, $K ( x _ { * } , X )$ is the $1 \times N$ cross-kernel vector between the test point and training data, λ is the ridge regularization penalty, acting as a noise floor to prevent overfitting. Generally, this is known as the KRR (Kernel ridge regression) problem [31].

KRR is solved in a two-step process utilizing dual coeficients. Instead of matrix inversion, we solve for the dual coeficient vector α:

$$
\left( K ( X , X ) + \lambda I \right) \alpha = y\tag{10}
$$

Once α is computed, inference for an arbitrary test set $X _ { \mathrm { t e s t } }$ becomes a highly eficient matrix-vector multiplication:

$$
f ( X _ { \mathrm { t e s t } } ) = K ( X _ { \mathrm { t e s t } } , X ) \alpha\tag{11}
$$

TABLE I : Comparison between NN results and Kernel $\mathrm { R e } { } \mathrm { - }$ gression: $T _ { c }$ is the prediction $T _ { c } ^ { 0 }$ , being the original data.
<table><tr><td>Method</td><td>mean  $| T _ { c } - T _ { c } ^ { 0 } |$ </td><td>rms  $( T _ { c } - T _ { c } ^ { 0 } ) ~ \bar { T } _ { c }$ </td></tr><tr><td>Neural Network</td><td>2.11K</td><td>7.20K 17.5K</td></tr><tr><td>Kernel Regression (trained)</td><td>2.24K</td><td>7.61K</td></tr><tr><td>Kernel Regression (untrained) 21.7K</td><td></td><td>31.9K</td></tr></table>

The comparable performance between the neural network and kernel ridge regression, shown in Table I, suggests that a substantial fraction of the learned representation is encoded in the trained NTK. In contrast, kernel regression using the NTK at random initialization performs substantially worse, indicating that the predictive structure captured by the trained NTK is not inherited from the architecture alone, but emerges during training through feature learning. This observation motivates treating the trained NTK not merely as a visualization tool, but as an interpretable representation of the learned predictor itself.

## IV. NEURAL TANGENT KERNEL DISTANCE AND FEATURE LEARNING

Given a trained neural network $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ , we define the neural tangent kernel (NTK) at the trained parameters $\theta ^ { * }$ as

$$
K ( x , x ^ { \prime } ) = \nabla _ { \boldsymbol { \theta } } f _ { \boldsymbol { \theta } } ( x ) ^ { \top } \nabla _ { \boldsymbol { \theta } } f _ { \boldsymbol { \theta } } ( x ^ { \prime } ) \Big | _ { { \boldsymbol \theta } = { \boldsymbol { \theta } } ^ { * } } .\tag{12}
$$

This kernel can be interpreted as an inner product in a feature space $\phi ( x ) = \nabla _ { \theta } f _ { \theta } ( x )$ . Based on this inner product, we define a distance:

$$
d ( x , x ^ { \prime } ) ^ { 2 } = K ( x , x ) + K ( x ^ { \prime } , x ^ { \prime } ) - 2 K ( x , x ^ { \prime } ) ,\tag{13}
$$

corresponding to Euclidean distance in the induced feature space. This geometry in the feature space is closely related to kernel PCA and related manifold learning methods, where low-dimensional structures are extracted from the spectral decomposition of similarity kernels [32, 33].

As Fig. 2(a) shows, the initial NTK largely preserves compositional distances inherited from the input representation. By contrast, as presented in $\mathrm { F i g ~ 2 ( d ) }$ , the trained NTK substantially reorganizes the geometry, bringing many compositionally distant but related materials closer together in feature space. This suggests that training induces a compression of materials space aligned with superconducting properties.

Classical NTK theory considers regimes in which the kernel remains fixed during training [21, 34]. However, in practical settings, the NTK evolves and becomes aligned with the target function [22, 23]. This reflects feature learning: the network adapts its representation so that relevant directions are amplified. As a result, the trained NTK often exhibits structured, lowrank spectra, with leading eigenspaces capturing taskrelevant variation [24].

Having established that training substantially reorganizes the geometry induced by the NTK, we now investigate the dimensional structure of this learned representation through spectral analysis. Our sample space is composed of $N _ { s } = 1 6 4 1 4$ data points, making the NTK an $N \times N$ object. The coordinates associated with the $x _ { i } \mathrm { s }$ can be thought of as independent degrees of freedom. To assess their importance, we examine the efect of regularization λ on the eigenspectrum of our problem. Conveniently, this corresponds to the estimate of the ridge regression on our kernel [35, 36]. We let,

(a)  
![](images/2cdb6b52ec3ec8ad988dce4a16b3cfe8bb4674989b09c1383d90abbe37fa3e4c.jpg)

(b)  
![](images/b7e3393565e1c4482d8a9c319b3ab5781a41a456e420b0fee3fdc5a9d4507195.jpg)

(c)  
![](images/ee2d77d307261649de0fa439a319553ce6a8e8609a18e3c549c82808a7025c87.jpg)

(d)  
![](images/9b957871a4349a640e04032e6c146b04cb6cd381255ab86de1e556ece8514075.jpg)  
FIG. 2 : Properties and structure of the NTK. (a) The Shepard plot, i.e., embedding distance $\vert \vert z _ { i } - z _ { j } \vert \vert$ vs Euclidean distance of an untrained NTK realization, derived first the first $N = 1$ step of training. No embedding is generated. (b) Eigenvalue spectrum of the train NTK. Approximately two principal eigenvalues can be discerned. (Note the log scale). (c) Kernel Ridge Regression (KRR) properties: rapid convergence of the KRR with respect to the regularization λ confirms the relative number of useful degrees of freedom (non-null eigenvalues in (b)). (d) Shepard plot of the trained NTK, indicating a significantly diferent embedding of the data.

$$
d _ { f } = \mathrm { T r } ( K ( K + \lambda I ) ^ { - 1 } ) ,\tag{14}
$$

where $d _ { f }$ is the proxy for the true degrees of freedom in the kernel. In Fig. 2(c), we plot $d _ { f }$ as a function of λ.

The trained kernel its spectral weight in a small number of leading directions. We quantify this concentration using the regularized efective dimension, $\begin{array} { r } { d _ { \mathrm { e f f } } ( \lambda ) = \sum _ { a } \kappa _ { a } / ( \kappa _ { a } + \lambda ) } \end{array}$ , and the fraction of centered feature variance retained by the leading MDS coordi nates. These measures distinguish the dominant geometric directions from the larger set of nonzero kernel modes.

This can be further strengthened by computing the eigenvalues exactly in the sample space $N _ { s }$ . In Fig. 2(b) we show the characteristic eigenvalue spectrum of the NTK, with $N _ { s } = 1 0 0 0 0$ (for visual clarity, we truncate all zero eigenvalues beyond this point), on a log scale of eigenvalue intensity. We find that a select number of features $N _ { f } < 1 0$ possess eigenvalues in the range of $1 0 ^ { 6 } - 1 0 ^ { 9 }$ This reveals the most significant features which the NTK is able to resolve without additional training. For this reason, this is a first glimpse into the possibility of extracting ”transferable” learning and a low-dimensional manifold which learned through the supervised training.

To visualize the structure encoded in the trained NTK distances, we construct a low-dimensional embedding using classical multi-dimensional scaling (MDS) [37]. Given the pairwise distance matrix $d ( x _ { i } , x _ { j } )$ defined in $\operatorname { E q . }$ . (5), classical MDS seeks coordinates $z _ { i } \in$ $\mathbb { R } ^ { K }$ whose Euclidean distances approximately preserve the original NTK geometry:

$$
| | z _ { i } - z _ { j } | | \approx d ( x _ { i } , x _ { j } ) .\tag{15}
$$

Equivalently, MDS constructs an embedding by diagonalizing the double-centered distance matrix. Concretely, let,

$$
\Pi = I - \frac { 1 } { N } { \bf 1 1 ^ { \top } } , ~ B = - \frac { 1 } { 2 } \Pi { \cal D } ^ { ( 2 ) } \Pi = \Pi K \Pi .\tag{16}
$$

Here, $D ^ { ( 2 ) }$ , producing a low-dimensional representation that preserves the dominant geometric relationships between our inputs, which are materials $( x _ { i } )$ . Applying this to the NTK distances from the SuperCon dataset, we find that even for $K = 2 .$ , the explained variance is 92%, and going up to $K = 3$ increases it marginally to 95.3%. This is consistent with the efective low-rank structure of the Kernel as implied by Figs. 2(b)-(c).

## V. NTK GEOMETRY OF MATERIALS SPACE

A central question, now, is whether this lowdimensional embedding captures physically meaningful structure or merely reflects a geometric compression of the data. To probe this question, we examine both the clustering structure induced in the embedded space and correlations with physically relevant observables not used directly in training. The emergence of coherent superconducting families together with correlations between diferent chemical families implies that the NTK embedding encodes meaningful structure within materials space rather than arbitrary geometric organization.

## A. Clustering the Latent Vectors

We apply k-means [38] to the latent vectors $\mathbf { z } _ { i }$ . We partition the data into C sets $\mathbf { S } = \{ S _ { 1 } , S _ { 2 } , \ldots , S _ { C } \}$ by minimizing the within-cluster variance:

$$
\arg \operatorname* { m i n } _ { \mathbf { S } } \sum _ { c = 1 } ^ { C } \sum _ { \mathbf { z } \in S _ { c } } \| \mathbf { z } - \pmb { \mu } _ { c } \| ^ { 2 }\tag{17}
$$

where $\pmb { \mu } _ { c }$ is the centroid of cluster $S _ { c }$

We apply this framework to superconducting materials data based on the SuperCon dataset [39]. We test several cluster sizes, searching for maximum explained variance, for the smallest number of clusters (see $\mathrm { A p \mathrm { - } }$ pendix). We find that $N = 8$ gives us the best uniform covering of the data. The resultant clusters are presented in Fig. 3, which is one of our main results. These clusters are generated without additional learning or supervision, from the matrix $d ( x , x ^ { \prime } )$ and with the application of the MDS algorithm.

TABLE II : Mean Silhouette Scores per Cluster
<table><tr><td>Cluster</td><td>Mean Silhouette</td></tr><tr><td>Mixed 1</td><td>0.175</td></tr><tr><td>High Entropy</td><td>0.415</td></tr><tr><td>High  $T _ { c }$ </td><td>0.450</td></tr><tr><td>Iron Based</td><td>0.350</td></tr><tr><td>Mixed 2</td><td>0.200</td></tr><tr><td>Mixed 3</td><td>0.140</td></tr><tr><td>Elemental/Simple</td><td>0.300</td></tr><tr><td>Non-SC</td><td>0.425</td></tr></table>

In addition, some clusters are dificult to label and do not contain an obvious preponderance from a specific chemical family. To check the reliability of this clustering, we inspect their relative silhouettes [40]. Inspecting Tab II, we see that the clusters with higher silhouette scores correspond to well-known superconducting material classes as well as the non-superconducting class. This is an indication that the structure of the latent space has learned some information about the underlying physics.

## B. Notion of Neighborhoods

To answer the question concerning the significance of distances presented by $d ( x _ { i } , x _ { j } )$ , we note that the

Shepard plots of Fig. 2 ofer a valuable observation: objects in the embedding space are often closer than the metric related to their input space, $\mathrm { i . e . , } d _ { \mathrm { e u c l i d e a n } } =$ $x _ { i } \cdot x _ { j } .$ To inspect this, we explicitly investigate the NTK distance, centered on select compounds. In order to normalize and define a bona fide distance metric, we normalize the NTK distance $d ( x _ { i } , x _ { j } )$ by the diagonal of the NTK Gram matrix, i.e., $\begin{array} { r } { d _ { i j } ^ { 2 }  \frac { d _ { i j } ^ { 2 } } { K ( x _ { i } , x _ { i } ) + K ( x _ { j } , x _ { j } ) } . } \end{array}$ Then, to illustrate the geometric structure of the space spanned by $d _ { i j }$ , which we refer to as “material space”, we take two archetypal compounds as origin and order the materials around it by their radial distance. This is akin to measuring pair correlation functions – a familiar practice in the physics of materials. $\mathrm { Y _ { 1 } B a _ { 1 . 9 2 } C a _ { 0 . 0 7 4 } C u _ { 4 } O _ { 8 } }$ is a famous high temperature superconductor with $T _ { c }$ ≈ 80 K, and CeCoIn<sub>5</sub> a heavy fermion superconductor with $T _ { c }$ 1.8 K.

In Fig. 4 (top) we show data for YBCO, and Fig. 4(bottom) describes CeCoIn . We observe several notable trends. Relative to the same distance, we plot $T _ { c }$ against the NTK distance. Remarkably, the distance appears uncorrelated with the variation in $T _ { c } .$ . Compounds with $d \to 0$ nevertheless show fluctuations of $\bar { \Delta } T _ { c } ~ \approx ~ \pm 2 0 K$ This astounding result is explained in part by the data itself: the NTK distance measure correctly identified doping, chemical similarity, and the structures of generic chemical formula classes (chemical families, such as the typical 1237, 113, etc).

Fig. 4 (bottom), we reveal an even sharper distinction between inputs, which is once more uncorrelated with any label in the supervised learning. To properly model “neighborhoods”, we introduce a notion of gaps in the distance: we separate regions based on $d > d _ { c } ,$ which is identified individually for every compound family. These gaps are plotted with a gray dashed line in Fig. 4.

Initially (within the nearest-neighbor) for CeCoIn , we find mostly doped versions of the same basic unit $( \mathrm { i . e . , ~ \ e . g . ~ \quad ~ C e _ { 0 . 9 8 } L a _ { 0 . 0 4 } \dots } )$ However, within the next-nearest neighbor cluster, we discover a cousin of $\mathrm { C e C o I n _ { 5 } } .$ the heavy fermion and strange metal system YbRh<sub>2</sub>Si<sub>2</sub>. The learned neighborhood connects CeCoIn with $\mathrm { Y b R h _ { 2 } S i _ { 2 } }$ although their compositional vectors have zero overlap because they share no constituent elements. Neither heavy-fermion behavior nor strange-metal behavior was supplied as a training label.

The figure demonstrates that the network has learned features significantly diferent from the task it was trained on, $T _ { c }$ . Materials with very diferent superconducting $T _ { c } ,$ but with other similarities such as similar crystal structure or generic chemical formula, or fundamental physical properties, such as strange metallicity or heavy-fermion ground states, are remarkably close by thanks to the NTK distance.

![](images/d1329465b7eff0d561bd06be459eb532b2e8e4a2e630239034b2b776e79f373b.jpg)  
FIG. 3 : Low-dimensional embedding of superconducting materials obtained from classical multidimensional scaling (MDS) applied to trained NTK distances. Colors correspond to clusters identified through unsupervised k-means clustering in the embedded representation space. Several clusters align with known superconducting families, including cuprates, iron-based superconductors, undoped or otherwise superconductors with “simple” chemical formulae, and non-superconducting compounds. As this information is not present during training, we manually assign labels to the generated clusters; each cluster taxonomy is determined by the the first 200 compounds, measured by distance to the centroid.

## C. Measuring uncertainty

So far, our distance measure $d _ { i j }$ avoided the element $K ( x _ { i } , x _ { i } )$ , since $d _ { i i } = 0$ by definition. Here we give a direct connection to the object $K ( x _ { i } , x _ { i } )$ and discuss its implications on the appearance of clusters in our trained network.

To connect this quantity to uncertainty, consider small, zero-mean parameter fluctuations with covariance $\langle \delta \theta _ { \mu } \delta \theta _ { \nu } \rangle = \sigma _ { \theta } ^ { 2 } \delta _ { \mu \nu }$ Within the tangent approximation, the resulting output fluctuation is $\delta f ( x ) \ =$ $\nabla _ { \theta } f _ { \theta _ { * } } ( x ) \cdot \delta \theta ;$ this can thought of as a connected twopoint function, which therefore satisfies $\begin{array} { r l } { G _ { \mathrm { c } } ( x , x ^ { \prime } ) } & { { } = } \end{array}$ $\langle \delta f ( x ) \delta f ( x ^ { \prime } ) \rangle = \sigma _ { \theta } ^ { 2 } K ( x , x ^ { \prime } )$ [41]. The coincident-point limit gives Var $[ f ( \bar { x } ) ] = G _ { \mathrm { c } } ( x , x ) = \sigma _ { \theta } ^ { 2 } K ( x , x )$ , directly paralleling the identification of local fluctuations with the diagonal of a connected two-point function in statistical field theory [42]. Accordingly, $K ( x , x )$ measures the output variance per unit isotropic parameter variance, while the associated uncertainty scale is $\sigma _ { f } ( x ) = \sigma _ { \theta } \sqrt { K ( x , x ) }$ . Larger diagonal values thus identify inputs whose predictions fluctuate more strongly under the same level of uncertainty in the learned parameters. This interpretation applies locally around the trained configuration, requires no infinite-width assumption, and is closely related to uncertainty constructions based on linearized networks and trained empirical NTKs [43]. In Fig. 5, we show a particular application to this uncertainty measure. We fix $z _ { 2 } = - 2 5$ and average over a window of ±6. In turn, we plot $K ( x , x )$ across the coordinate $z _ { 1 }$ . We find an increase in $K ( x , x )$ precisely at the boundaries of well-identified clusters, and in areas where diferent samples from diferent clusters leak into each other. Such an identifiable behavior was not established in areas of poor explanability and correspondingly low silhouette.

## VI. DISCUSSION

In this work, we have shown that the trained neural tangent kernel provides an interpretable probe of the representation learned by a supervised neural network. Using superconducting materials as a model system, we demonstrated that the trained NTK induces a low-dimensional geometry that faithfully reproduces the predictive capability of the original network while organizing materials according to physically meaningful rela tionships. Unlike the NTK at random initialization, the trained kernel substantially reorganizes the geometry inherited from the input representation, revealing neighborhoods, clusters, and dominant directions associated with the learned task. Importantly, these structures are recovered without retraining the network or introducing additional representation-learning objectives.

![](images/619a4e413e72da3bea5f85c31ed77645e39614dd1ce7e521532578472e97fd8b.jpg)  
FIG. 4 : Neighborhood structure induced by the trained NTK geometry. Materials are ordered according to radial distance from representative compounds in the learned NTK metric (top: $\mathrm { Y B a _ { 2 } C u _ { 4 } O _ { 8 } } .$ -based cuprate; bottom: CeCoIn heavy-fermion superconductor). Blue points show NTK distance, while red points indicate the superconducting transition temperature $T _ { c }$ Gray dashed lines denote distance gaps used to identify neighborhood shells in the learned geometry. Nearby compounds frequently share chemical motifs, doping structure, or correlated-electron phenomenology despite substantial variation in $T _ { c } ,$ indicating that the trained NTK captures transferable physical structure beyond the original supervised learning task.

More broadly, our work complements the long tradition of representation learning in machine learning. Classical approaches such as independent component analysis, sparse coding, and more recent self-supervised methods explicitly optimize latent representations from data. Our work addresses the complementary problem of representation recovery rather than representation learning. Once a supervised predictor has been successfully trained, we ask what scientific representation it has already acquired and how it can be interpreted. Our results suggest that the trained NTK provides a practical means of recovering this representation without auxiliary objectives or additional optimization.

Perhaps the most surprising aspect of our results is that this representation emerges from elemental composition alone. As we stated in the introduction, popular approaches to generate a meaningful low-dimensional manifold require an abundance of features fed during training. In contrast, the predictor considered here receives only elemental composition of materials. The emergence of a meaningful representation under these conditions suggests that elemental composition itself contains a surprisingly rich latent organization. Supervised learning then uncovers this organization, and the trained NTK provides a means of recovering and interpreting it.

![](images/137442b7ebdb3e9e501e6837c6074c2ab66c9e8d9d7603b5f5666246c03ab754.jpg)  
FIG. 5 : Measure of $K ( x , x )$ as uncertainty in clustering. We take a cut along a fixed $z _ { 2 }$ value $z _ { 2 } = - 2 5$ and average over a window of ±6 in that range. We then traverse $z _ { 1 }$ . We find a correlation between boundaries of the identified clusters and increases in $K ( x , x )$ . In areas of low explainability/silhouette, such corresponding changes in $K ( x , x )$ were not observed.

More broadly, our results suggest that trained tangent kernels should be viewed not only as analytical tools for understanding optimization, but also as scientific objects in their own right. Their induced geometry provides an interpretable representation of learned scientific structure, ofering a bridge between modern representation learning, kernel methods, and the analysis of complex materials datasets.

[1] P. Reiser, M. Neubert, A. Eberhard, L. Torresi, C. Zhou, C. Shao, H. Metni, C. van Hoesel, H. Schopmans, T. Sommer, and P. Friederich, Graph neural networks for materials science and chemistry, Communications Materials 3, 10.1038/s43246-022-00315-6 (2022).

[2] S. M. Moosavi, K. M. Jablonka, and B. Smit, The role of machine learning in the understanding and design of ma-

## ACKNOWLEDGMENTS

We thank D. Chowdhury, P. de Faira, P. Ginsparg, E. Kim, O. Lesser and T. Scafidi for useful discussions. D.K. acknowledges the support of the Abrahams postdoctoral fellowship of the Center for Materials Theory, Rutgers University and the Zuckerman STEM Fellowship. This work as completed in part at Aspen Center for Physics, which is supported by National Science Foundation grant PHY-2210452.

terials, Journal of the American Chemical Society 142, 20273 (2020).

[3] J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, How transferable are features in deep neural networks?, Advances in neural information processing systems 27 (2014).

[4] D. Buterez, J. P. Janet, S. Kiddle, D. Oglic, and

P. Li\`o, Transfer learning with graph neural networks for improved molecular property prediction in the multi-fidelity setting, ChemRxiv 10.26434/chemrxiv-2022-dsbm5-v3 (2024).

[5] S. R¨ocken and J. Zavadlav, Enhancing machine learning potentials through transfer learning across chemical elements (2025), arXiv:2502.13522 [physics.chem-ph].

[6] A. Arzani, L. Yuan, P. Newell, and B. Wang, Interpreting and generalizing deep learning in physics-based problems with functional linear models, Engineering with Computers 41, 135 (2025).

[7] Y. LeCun, L. Bottou, Y. Bengio, and P. Hafner, Gradient-based learning applied to document recognition, Proceedings of the IEEE 86, 2278 (1998).

[8] K. He, X. Zhang, S. Ren, and J. Sun, Deep residual learning for image recognition, in Proceedings of the IEEE conference on computer vision and pattern recognition (2016) pp. 770–778.

[9] A. J. Bell and T. J. Sejnowski, The “independent components” of natural scenes are edge filters, Vision research 37, 3327 (1997).

[10] B. A. Olshausen and D. J. Field, Emergence of simplecell receptive field properties by learning a sparse code for natural images, Nature 381, 607 (1996).

[11] Y. Bengio, A. Courville, and P. Vincent, Representation learning: A review and new perspectives, IEEE transactions on pattern analysis and machine intelligence 35, 1798 (2013).

[12] Y. Suzuki, T. Taniai, K. Saito, Y. Ushiku, and K. Ono, Self-supervised learning of materials concepts from crystal structures via deep neural networks, Machine Learning: Science and Technology 3, 045034 (2022).

[13] J. Z. Kim, O. Lesser, and D. Chowdhury, Charting the emergent low-dimensional manifold of quantum materials (2026), arXiv:2606.12520 [cond-mat.supr-con].

[14] R. Okabe, M. Cheng, A. Chotrattanapituk, M. Mandal, K. Mak, D. Cordova Carrizales, N. T. Hung, X. Fu, B. Han, Y. Wang, et al., Structural constraint integration in a generative model for the discovery of quantum materials, Nature Materials 25, 223 (2026).

[15] A. Panigrahi, Y. Liu, O. Lesser, K. Mallayya, and E.- A. Kim, Graphlet histogram representation database of inorganic crystals (2026), arXiv:2606.10195 [condmat.mtrl-sci].

[16] D. Kaplan, A. Zheng, J. Blawat, R. Jin, R. J. Cava, V. Oudovenko, G. Kotliar, A. M. Sengupta, and W. Xie, Deep learning-based superconductivity prediction and experimental tests, The European Physical Journal Plus 140, 58 (2025).

[17] L. Ward, A. Agrawal, A. Choudhary, and C. Wolverton, A general-purpose machine learning framework for predicting properties of inorganic materials, npj Computational Materials 2, 16028 (2016).

[18] V. Stanev, C. Oses, A. G. Kusne, E. Rodriguez, J. Paglione, S. Curtarolo, and I. Takeuchi, Machine learning modeling of superconducting critical temperature, npj Computational Materials 4, 29 (2018).

[19] T. Xie and J. C. Grossman, Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties, Physical review letters 120, 145301 (2018).

[20] K. Choudhary and B. DeCost, Atomistic line graph neural network for improved materials property predictions, npj Computational Materials 7, 185 (2021).

[21] A. Jacot, F. Gabriel, and C. Hongler, Neural tangent kernel: Convergence and generalization in neural networks, Advances in neural information processing systems 31 (2018).

[22] S. Fort, G. K. Dziugaite, M. Paul, S. Kharaghani, D. M. Roy, and S. Ganguli, Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the neural tangent kernel, Advances in Neural Information Processing Systems 33, 5850 (2020).

[23] H. Shan and B. Bordelon, A theory of neural tangent kernel alignment and its influence on training, arXiv preprint arXiv:2105.14301 (2021).

[24] A. Canatar and C. Pehlevan, A kernel analysis of feature learning in deep neural networks, in 2022 58th Annual Allerton Conference on Communication, Control, and Computing (Allerton) (IEEE, 2022) pp. 1–8.

[25] P. Gegenwart, Q. Si, and F. Steglich, Quantum criticality in heavy-fermion metals, nature physics 4, 186 (2008).

[26] C. Petrovic, P. Pagliuso, M. F. Hundley, R. Movshovich, J. L. Sarrao, J. D. Thompson, Z. Fisk, and P. Monthoux, Heavy-fermion superconductivity in cecoin5 at 2.3 k, Journal of Physics: Condensed Matter 13, L337 (2001).

[27] B. B. Zhou, S. Misra, E. H. da Silva Neto, P. Aynajian, R. E. Baumbach, J. Thompson, E. D. Bauer, and A. Yazdani, Visualizing nodal heavy fermion superconductivity in cecoin5, Nature physics 9, 474 (2013).

[28] The value 120 here is meant to represent the number of elements in the periodic table, padded with two more to allow the factorization 12 × 10 for an image input to the CNN.

[29] P. W. Koh and P. Liang, Understanding black-box predictions via influence functions, in Proceedings of the 34th International Conference on Machine Learning (PMLR, 2017) pp. 1885–1894.

[30] D. S. Bernstein, Scalar, vector, and matrix mathematics: theory, facts, and formulas-revised and expanded edition (Princeton university press, 2018).

[31] D. G. Krige, A statistical approach to some basic mine valuation problems on the witwatersrand, Journal of the Southern African Institute of Mining and Metallurgy 52, 119 (1951).

[32] B. Sch¨olkopf, A. Smola, and K.-R. M¨uller, Nonlinear component analysis as a kernel eigenvalue problem, Neu ral computation 10, 1299 (1998).

[33] R. R. Coifman and S. Lafon, Difusion maps, Applied and computational harmonic analysis 21, 5 (2006).

[34] S. Du, J. Lee, H. Li, L. Wang, and X. Zhai, Gradient descent finds global minima of deep neural networks, in International conference on machine learning (PMLR, 2019) pp. 1675–1685.

[35] V. Vovk, Kernel ridge regression, in Empirical inference: Festschrift in honor of vladimir n. vapnik (Springer, 2013) pp. 105–116.

[36] X. Chang, S.-B. Lin, and D.-X. Zhou, Distributed semisupervised learning with kernel ridge regression, Journal of Machine Learning Research 18, 1 (2017).

[37] I. Borg and P. J. Groenen, Modern multidimensional scaling: Theory and applications (Springer, 2005).

[38] J. B. McQueen, Some methods of classification and analysis of multivariate observations, in Proc. of 5th Berkeley Symposium on Math. Stat. and Prob. (1967) pp. 281–

297.

[39] M. NIMS, Superconducting material database (supercon), 10.48505/nims.3739 (2022).

[40] P. J. Rousseeuw, Silhouettes: a graphical aid to the interpretation and validation of cluster analysis, Journal of computational and applied mathematics 20, 53 (1987).

[41] A. Immer, M. Korzepa, and M. Bauer, Improving predictions of Bayesian neural nets via local linearization, in Proceedings of the 24th International Conference on Artificial Intelligence and Statistics, Proceedings of Ma-

chine Learning Research, Vol. 130 (PMLR, 2021) pp. 703–711.

[42] M. Kardar, Statistical Physics of Fields (Cambridge University Press, Cambridge, 2007).

[43] J. Wilson, C. van der Heide, L. Hodgkinson, and F. Roosta, Uncertainty quantification with the empirical neural tangent kernel, in Advances in Neural Information Processing Systems, Vol. 38 (2025).