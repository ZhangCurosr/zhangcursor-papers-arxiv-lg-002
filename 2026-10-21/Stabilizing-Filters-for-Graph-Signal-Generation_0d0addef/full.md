# Stabilizing Filters for Graph Signal Generation

Martin Schmidt and Gonzalo Mateos

Abstract—Generating graph signals is an important task in machine learning, and incorporating the underlying graph into a generative model enables structurally-informed distribution modeling. However, said graph is often noisy or imperfectly estimated, and structural errors can affect the generative process and the resulting signal distribution. While recent work has established stability guarantees for graph neural ordinary differential equations, the robustness of graph-aware reference dynamics, as used in topological Schrodinger bridge models, remains¨ largely unexplored. Here, we investigate the structural stability of continuous-time generative models whose dynamics combine a graph filter with a learned graph neural network. We first derive a Wasserstein stability bound that quantifies the effect of graph perturbations on the generated signal distributions. Informed by this analysis, we then propose a principled framework for designing graph filters that balance structural stability with induced signal smoothing. We characterize the optimal filter design and provide an explicit family of polynomial filters with provable stability guarantees. Experiments on synthetic graphs and real-world fMRI signals support our theoretical findings, illustrating the enhanced structural stability of the proposed filters while preserving generative performance.

Index Terms—Graph signal processing, generative models, Schrodinger bridges, stability, permutation equivariance.¨

## I. INTRODUCTION

Finding a map transporting a given data distribution to another is a fundamental problem underlying generative modeling, with numerous applications in machine learning and signal processing; see e.g., [2], [3]. Recent advances in continuoustime generative modeling [4]–[6] have achieved remarkable success generating realistic images, audio and video. However, their application to irregular, non-Euclidean domains that can be naturally represented using graphs is still fairly underdeveloped. While considerable research has focused on generating graph topologies [7]–[12], far less attention has been paid to generating graph signals [13]–[17]. These signals are ubiquitous in real-world networked systems, such as neural time courses over structural brain connectomes [18], vehicular flows on transportation networks [19], and resource (e.g., power) allocations in wireless networks [20]. Learning to faithfully generate graph signals has broad applicability, ranging from synthetic data augmentation and implicit priors to realistic system simulations across various timely domains.

Graph-aware reference dynamics. Among the many stochastic processes that match a given pair of endpoint distributions, the dynamic Schrodinger bridge problem provides a principled¨ solution that favors the process closest to a prescribed reference dynamics [21]–[23]. In the classical formulation, the reference is typically chosen as a standard Wiener process. However, when transporting distributions of graph signals, it is prudent to infuse the underlying structure directly into the generative model. Adopting custom graph-aware reference dynamics (e.g., heat diffusion over the graph) [15]–[17] promotes trajectories that are consistent with the graph topology.

Permutation equivariance and stability. By design, these generative models depend on the underlying graph to parametrize both the reference dynamics and the learned vector field. In practice, however, the observed graph is only an imperfect reflection of the true complex system [24, Ch. 7]. For instance, correlation networks inferred from nodal measurements can vary significantly with the sample size [25], leading to noisy graph representations [26], [27]. Consequently, unavoidable graph perturbations can skew the entire generative trajectory and the final distribution of the generated signals. This motivates the study of structural stability, which quantifies the sensitivity of generative model outputs to perturbations in the graph parameterizing the generative dynamics.

For graph-structured data, permutation equivariance is a desired property to ensure that the generative dynamics natively respect graph-induced symmetries [28]. Furthermore, because evaluating robustness across graphs with arbitrary node indexing calls for stability notions modulo permutation [29], equivariance is a mathematical prerequisite for any meaningful structural stability analysis, rather than merely a useful prior.

Innovations in context. While permutation equivariance, structural stability, and transferability have been rigorously established for graph neural networks (GNNs) [29]–[33], these properties have primarily been studied in the context of static, discriminative tasks. In contrast, continuous-time generative models evolve a state trajectory that maps one distribution into another. Recent efforts have begun to bridge this gap, establishing structural stability [34] and transferability [35] bounds for graph neural ordinary differential equations (ODEs), i.e., GNN-parametrized ODEs [36]. However, while the stability of a learned vector field can be controlled via regularization during training [34, Sec. V], graph-aware reference dynamics rely on a prescribed graph filter [15]–[17]. This raises the question of how to properly design stable reference dynamics.

In this context, our significant contributions to the foundations of graph signal generation are twofold: (i) we characterize the structural stability of generative models driven by graph-aware reference dynamics, deriving Wasserstein bounds that elucidate how graph perturbations affect the generated distributions; and (ii) we introduce an optimization framework to design graph filters that respect the induced smoothing of heat diffusion, while boosting robustness to graph imperfections.

Paper outline. The remainder of the paper is organized as follows. Section II reviews the necessary background on graph signal processing (GSP), the structural stability of graph operators, and generative modeling with topological Schrodinger bridges; as needed to formally state the prob-¨ lem. Section III formulates the continuous-time generative dynamics, and derives Wasserstein stability bounds under structural perturbations. Leveraging these theoretical insights, Section IV introduces our principled filter design framework to delineate relevant stability-smoothness tradeoffs. To this end, we formulate a semi-infinite linear program and derive explicit optimal polynomial filter constructions. Section V numerically corroborates the efficacy of our optimal filter design relative to the heat diffusion baseline, both in a synthetic test case and a study involving fMRI brain signals. Finally, Section VI concludes the paper by summarizing our findings, discussing the impact of the chosen perturbation model on the optimal design, and contextualizing the practical upshot of our bounds. All technical details and proofs are deferred to the appendices.

Relative to the conference precursor [1], here we consider stabilizing graph filters for generative modeling through a unified presentation along with full technical details (including extended discussions and full proofs), supported by comprehensive numerical experiments. Noteworthy novel contributions in this journal paper include: (i) the full proofs of all theoretical results; (ii) the direct optimization of the exact integral Lipschitz constant, which departs from the surrogate bound minimized in [1]; (iii) a rigorous theoretical characterization of the optimal filter convergence, including an explicit analytical polynomial construction that approaches the lower bound at an exponential rate; and (iv) a formal guarantee that optimality is attainable by a filter of finite order.

## II. PRELIMINARIES AND PROBLEM STATEMENT

Here we review key GSP concepts and survey GNN stability results, followed by the fundamentals of Schrodinger bridges¨ for graph-supported data. We close with a problem statement.

## A. Graph Signal Processing

Let $\mathcal { G } = ( \nu , \mathcal { E } , \mathcal { W } )$ be a known undirected graph with N nodes, edge set $\mathcal { E } \subseteq \mathcal { V } \times \mathcal { V }$ , and edge weight map $\mathcal { W } : \mathcal { E }  \mathbb { R }$ Its structure is represented by a graph shift operator (GSO) $\mathbf { L } \in \mathbb { R } ^ { N \times N }$ , e.g., the graph Laplacian. A graph signal $x :$ $\nu \to \mathbb { R }$ is represented by a vector $\mathbf { x } \in \mathbb { R } ^ { N }$ , with $x _ { i }$ denoting the signal value at node i. By the spectral theorem, the symmetric GSO admits the eigendecomposition $\mathbf { L } = \mathbf { V } \mathbf { A } \mathbf { V } ^ { \top }$ with eigenvalues $\{ \lambda _ { i } \} _ { i = 1 } ^ { N }$ and orthonormal eigenvectors $\mathbf { V }$ Accordingly, the graph Fourier transform (GFT) of x is $\hat { \mathbf { x } } =$ $\mathbf { V } ^ { \top } \mathbf { x } [ 3 7 ] . \mathrm { ~ A ~ }$ graph convolutional filter of order $P$ is defined as $\begin{array} { r } { \mathbf { F } ( \mathbf { L } ) : = \bar { \sum } _ { p = 0 } ^ { P } \theta _ { p } \mathbf { L } ^ { p } } \end{array}$ , with coefficients $\{ \theta _ { p } \} ~ [ 3 8 ]$ , [39], and corresponding frequency response $\begin{array} { r } { f ( \lambda ) : = \sum _ { p = 0 } ^ { P } \theta _ { p } \lambda ^ { p } } \end{array}$ A GNN extends this construction by cascading compositions of graph filters with pointwise nonlinearities $\sigma ( \cdot ) ~ [ 4 0 ] { - } [ 4 2 ]$ For an input $\mathbf { x } ,$ we denote by $u _ { \boldsymbol { \theta } } ( \mathbf { x } ; \mathbf { L } )$ the output of a GNN with learnable parameters $\pmb { \theta }$ and fixed graph support L. For convenience, we henceforth use $\mathcal { A } ( \mathbf { x } ; \mathbf { L } )$ to denote a generic graph-supported operator, say a graph filter or a GNN acting on x, and parameterized by the GSO L.

## B. Stability of Graph Filters and GNNs

Let P denote the set of $N \times N$ permutation matrices. A desirable cardinal property of graph-supported operators is permutation equivariance [29], meaning

$$
\mathcal { A } ( \mathbf { P } ^ { \top } \mathbf { x } ; \mathbf { P } ^ { \top } \mathbf { L } \mathbf { P } ) = \mathbf { P } ^ { \top } \mathcal { A } ( \mathbf { x } ; \mathbf { L } )
$$

for any permutation $\mathbf { P } \in \mathcal { P }$ . In words, this means that $\mathrm { e . g . }$ GNN outputs are independent of arbitrary node labelings.

To analyze the robustness of graph filters and GNNs to structural errors, we adopt a well-documented relative perturbation model [29], [43]. Given a nominal GSO L and a perturbed one $\widetilde { \mathbf { L } } ,$ we evaluate the symmetric relative error matrix $\mathbf { E } \in \mathcal { S } : = \{ \mathbf { S } \in \mathbb { R } ^ { N \times N } : \mathbf { S } = \mathbf { \bar { S } } ^ { \top } \}$ at the permutation $\mathbf { P } _ { 0 } \in \mathcal { P }$ that minimizes the error’s spectral norm, i.e.,

$$
\begin{array} { l } { \displaystyle \{ { \bf E } ^ { \star } , { \bf P } _ { 0 } \} = \ \underset { { \bf E } \in { \cal S } , { \bf P } \in { \cal P } } { \mathrm { a r g m i n } } \ { \| { \bf E } \| _ { 2 } } } \\ { \displaystyle \mathrm { s . t . } \quad { \bf P } ^ { \top } \widetilde { { \bf L } } { \bf P } = { \bf L } + \frac { 1 } { 2 } \left( { \bf E } { \bf L } + { \bf L } { \bf E } \right) . } \end{array}\tag{1}
$$

Unlike absolute perturbations where edge corruptions are uniform, this relative perturbation model ensures that regions of $\mathcal { G }$ with weaker connectivity experience proportionally smaller changes compared to others with stronger links; see also [43, Sec. IV-A] for a detailed discussion further justifying the practical relevance of this widely adopted model.

We next introduce the integral Lipschitz condition associated with graph filter stability under relative perturbations.

Definition 1. Given a graph filter F with frequency response $f : \mathcal { D }  \mathbb { R }$ , where $\mathcal { D } \subseteq \mathbb { R }$ , we say the filter is integral Lipschitz if there exists a constant $C \geq 0$ such that

$$
| f ( \lambda _ { 2 } ) - f ( \lambda _ { 1 } ) | \leq 2 C { \frac { | \lambda _ { 2 } - \lambda _ { 1 } | } { | \lambda _ { 1 } + \lambda _ { 2 } | } } ,
$$

for all $\lambda _ { 1 } , \lambda _ { 2 } \in \mathcal { D }$ such that $\lambda _ { 1 } + \lambda _ { 2 } \neq 0 .$

For a differentiable frequency response $f ,$ taking the limit as $\lambda _ { 2 }  \lambda _ { 1 } = \lambda$ in Definition 1 implies that $| \lambda f ^ { \prime } ( \lambda ) | \leq C$ . The smallest admissible constant is the so-termed integral Lipschitz constant of $f ,$ denoted $C _ { \mathrm { I L } } ( f )$ , and is equivalently given by

$$
C _ { \mathrm { I L } } ( f ) : = \operatorname* { s u p } _ { \lambda _ { 1 } , \lambda _ { 2 } \in \mathcal { D } } \frac { | \lambda _ { 1 } + \lambda _ { 2 } | } { 2 } \frac { | f ( \lambda _ { 2 } ) - f ( \lambda _ { 1 } ) | } { | \lambda _ { 2 } - \lambda _ { 1 } | } .\tag{2}
$$

Graph filters and GNNs are structurally stable under the relative perturbation model (1). Specifically, for a relative perturbation satisfying $\| \mathbf { E } ^ { \star } \| _ { 2 } ~ \leq ~ \varepsilon$ and its corresponding optimal permutation $\mathbf { P } _ { 0 } .$ the pointwise deviation of a graphsupported operator satisfies

$$
\left\| \mathbf { P } _ { 0 } ^ { \top } A ( \mathbf { x } ; \widetilde { \mathbf { L } } ) - A ( \mathbf { P } _ { 0 } ^ { \top } \mathbf { x } ; \mathbf { L } ) \right\| \leq \left( \Gamma \varepsilon + \mathcal { O } ( \varepsilon ^ { 2 } ) \right) \| \mathbf { x } \| ,
$$

for all $\textbf { x } \in \ \mathbb { R } ^ { N }$ . The structural stability constant $\Gamma \ > \ 0$ dictates the sensitivity of the operator to structural noise. For a standalone graph filter F with frequency response $f ,$ the corresponding stability constant is

$$
\Gamma _ { F } = C _ { \mathrm { I L } } ( f ) \left( 1 + \delta \sqrt { N } \right) ,
$$

where $\delta \leq 8$ represents the eigenvector misalignment between $\mathbf { E } ^ { \star }$ and the nominal GSO L [29, Thm. 2]. Crucially, the integral Lipschitz constant $C _ { \mathrm { I L } } ( f )$ is the sole factor in this bound dictated by the filter design, whereas δ depends entirely on the realized structural perturbation. For a GNN, however, Γ also depends on the network’s depth, width, and the uniform spectral bounds of its intermediate filters [29, Thm. 4]. We build on these foundations to analyze the propagation of structural errors through the continuous-time dynamics of the following graph-aware generative model.

## C. Topological Schrodinger Bridge¨

Let M denote the set of probability measures on the path space $\mathcal { C } ( [ 0 , 1 ] , \mathbb { R } ^ { N } )$ . For any path measure $\mathbb { P } \in { \mathfrak { M } }$ , we denote by $\mathbb { P } _ { t }$ its time-t marginal distribution; $\mathrm { i . e . , i f } \ \mathbf { x } = ( \mathbf { x } _ { t } ) _ { 0 \leq t \leq 1 } \sim$ P, then $\mathbf { x } _ { t } \sim \mathbb { P } _ { t }$ . For prescribed initial and terminal marginals $\mathbf { x } _ { 0 } \sim p _ { 0 }$ and $\mathbf { x } _ { 1 } \sim p _ { 1 }$ , and given a reference path measure $\mathbb { Q } \in { \mathfrak { M } }$ , the Schrodinger bridge problem [44] seeks¨

$$
\mathbb { P } ^ { \star } \in \arg \operatorname* { m i n } _ { \mathbb { P } \in \mathfrak { M } } \mathrm { K L } ( \mathbb { P } \| \mathbb { Q } ) \quad \mathrm { s . t . } \quad \mathbb { P } _ { 0 } = p _ { 0 } , \ \mathbb { P } _ { 1 } = p _ { 1 } ,\tag{3}
$$

where $\mathrm { K L } ( \mathbb { P } | | \mathbb { Q } )$ denotes the Kullback–Leibler divergence between the path measures P and $\mathbb { Q } .$

For graph-supported data, the reference process can be chosen to explicitly incorporate the underlying topology [16]. Given $\mathrm { G S O ~ L , }$ consider the graph-aware reference diffusion

$$
d \mathbf { y } _ { t } = \left[ \mathbf { H } _ { t } ( \mathbf { L } ) \mathbf { y } _ { t } + \pmb { \alpha } _ { t } \right] d t + \pmb { \Sigma } _ { t } d \mathbf { w } _ { t } ,\tag{4}
$$

where $\mathbf { w } _ { t }$ is a standard Brownian motion in $\mathbb { R } ^ { N } , \Sigma _ { t } \in \mathbb { R } ^ { N \times N }$ controls the diffusion, ${ \pmb { \alpha } } _ { t } \in \mathbb { R } ^ { N }$ is a bias term, and $\mathbf { H } _ { t } ( \mathbf { L } ) : =$ $\scriptstyle \sum _ { k = 0 } ^ { K } h _ { k } ( t ) \mathbf { L } ^ { k }$ is a time-dependent graph filter of order $K .$ The reference path measure $\mathbb { Q }$ is then the law induced by (4).

Typically, samples are all we have from the endpoint distributions $p _ { 0 }$ and $p _ { 1 }$ . We consider generative dynamics governed by a stochastic differential equation (SDE) of the form

$$
d \mathbf { x } _ { t } = b _ { t } ( \mathbf { x } _ { t } ; \mathbf { L } ) d t + \Sigma _ { t } d \mathbf { w } _ { t } , \qquad \mathbf { x } _ { 0 } \sim p _ { 0 } ,\tag{5}
$$

where

$$
b _ { t } ( \mathbf { x } ; \mathbf { L } ) : = \mathbf { H } _ { t } ( \mathbf { L } ) \mathbf { x } + \pmb { \alpha } _ { t } + u _ { t } ^ { \pmb { \theta } } ( \mathbf { x } ; \mathbf { L } ) .\tag{6}
$$

Here, $u _ { t } ^ { \pmb \theta } ( \cdot ; \mathbf { L } )$ is a learned corrective vector field. It is trained so that the path law induced by (5) approximates the Schrodinger bridge ¨ $\mathbb { P } ^ { \star }$ in (3). Specifically, we parameterize this time-dependent vector field as $u _ { t } ^ { \pmb \theta } ( \mathbf { x } ; \mathbf { L } ) : = u _ { \pmb \theta } ( g ( \mathbf { x } , t ) ; \mathbf { L } )$ where $u _ { \pmb { \theta } } ( \cdot ; \mathbf { L } )$ is a GNN and $g ( \mathbf { x } , t )$ is a continuous temporal conditioning map that incorporates the time variable $t \in [ 0 , 1 ]$ into the signal; see [34, Sec. III] for a detailed discussion. Different training procedures can be used to estimate this vector field from samples, including Schrodinger bridge matching¨ and flow matching methods [16], [17].

## D. Problem Formulation

Given a trained generative model mapping the initial distribution $p _ { 0 }$ to the target distribution $p _ { 1 }$ via the dynamics in $( 5 ) – ( 6 )$ , our objective is to formally establish its robustness to graph perturbations. Let $\mathbf { L }$ denote the nominal GSO and $\widetilde { \mathbf { L } }$ be a perturbed GSO satisfying the relative perturbation model (1) bounded by $\| \mathbf { E } ^ { \star } \| _ { 2 } \leq \varepsilon$ . Since node indexing is arbitrary, we must compare the nominal and perturbed dynamics modulo permutations. Our twofold goal is to: (i) bound the Wasserstein distance between the resulting generated distributions in terms of $\varepsilon ;$ and (ii) extract actionable insights from said bound.

## III. STABILITY PROPERTIES

Let $\Phi _ { t } ( \mathbf { x } _ { 0 } ; \mathbf { L } )$ denote the solution of (5) at time t with initial condition $\mathbf { x } _ { 0 } ~ \sim ~ p _ { 0 }$ . For a permutation $\textbf { P } \in \textbf { \mathcal { P } }$ , we define the time-t marginal distributions of the permuted nominal and perturbed generative dynamics as

$$
\Phi _ { t } ( \mathbf { P } ^ { \top } \mathbf { x } _ { 0 } ; \mathbf { L } ) \sim \pi _ { t } ^ { \mathbf { P } } , \quad \mathbf { P } ^ { \top } \Phi _ { t } ( \mathbf { x } _ { 0 } ; \widetilde { \mathbf { L } } ) \sim \widetilde { \pi } _ { t } ^ { \mathbf { P } } .\tag{7}
$$

To ensure the generative dynamics respect the graph’s structural symmetries, the governing SDE must be permutation equivariant. The graph filter ${ \bf H } _ { t } ( { \bf L } )$ and the GNN $u _ { \pmb \theta } ( \cdot ; \mathbf { L } )$ are permutation equivariant by design (cf. Section II-B). To extend this property to the time-dependent vector field $u _ { t } ^ { \theta }$ , we assume that the temporal conditioning map $g$ is permutation equivariant, satisfying

$$
g ( \mathbf { P } ^ { \top } \mathbf { x } , t ) = \mathbf { P } ^ { \top } g ( \mathbf { x } , t ) , \qquad { \mathrm { f o r ~ a l l ~ } } \mathbf { P } \in { \mathcal { P } } .
$$

Together with the permutation equivariance of the GNN, this implies that the time-dependent vector field is itself permutation equivariant, namely,

$$
u _ { t } ^ { \theta } \big ( \mathbf { P } ^ { \top } \mathbf { x } ; \mathbf { P } ^ { \top } \mathbf { L } \mathbf { P } \big ) = \mathbf { P } ^ { \top } u _ { t } ^ { \theta } ( \mathbf { x } ; \mathbf { L } ) , \qquad \mathrm { f o r ~ a l l ~ } \mathbf { P } \in \mathcal { P } ,
$$

which follows directly from $u _ { t } ^ { \theta } ( \mathbf { x } ; \mathbf { L } ) : = u _ { \theta } ( g ( \mathbf { x } , t ) ; \mathbf { L } )$ . The assumed equivariance of $g$ is satisfied in practice by applying a temporal embedding (e.g., via node-wise addition or feature concatenation) uniformly across all nodes; see [34, Sec. III]. Such ways of incorporating temporal embeddings are widely used in generative models; see e.g., [45], [46].

We further assume that, for all $\mathbf { P } \in \mathcal { P }$ and $t \in [ 0 , 1 ]$ , the bias and diffusion satisfy $\mathbf { P } ^ { \top } \pmb { \alpha } _ { t } = \pmb { \alpha } _ { t }$ (e.g., a uniform drift ${ \pmb { \alpha } } _ { t } = c { \bf 1 } )$ and $\mathbf { P } ^ { \top } \pmb { \Sigma } _ { t } \mathbf { P } \overset { \cdot } { = } \pmb { \Sigma } _ { t }$ (e.g., isotropic noise $\Sigma _ { t } = \sigma _ { t } \mathbf { I } )$ Moreover, graph filters and GNNs are Lipschitz with respect to their input signals [34, Prop. 3]. Hence, for fixed $\mathbf { L } ,$ the drift $b _ { t } ( \cdot ; \mathbf { L } )$ is Lipschitz continuous, ensuring well-posedness and uniqueness of the generative SDE. Under these conditions, node relabeling commutes with the stochastic dynamics.

Proposition 1. For any initial distribution $\mathbf { x } _ { 0 } ~ \sim ~ p _ { 0 }$ , the probability laws of the generated trajectories $o f \ ( 5 )$ satisfy

$$
\begin{array} { r } { \mathbf { P } ^ { \top } \Phi _ { t } ( \mathbf { x } _ { 0 } ; \mathbf { L } ) \stackrel { d } { = } \Phi _ { t } ( \mathbf { P } ^ { \top } \mathbf { x } _ { 0 } ; \mathbf { P } ^ { \top } \mathbf { L } \mathbf { P } ) , } \end{array}
$$

for all $\mathrm { ~ \bf ~ P ~ } \in \mathrm { ~ \bf ~ \mathcal ~ P ~ }$ and $t ~ \in ~ [ 0 , 1 ]$ , where $\circeq$ means equal in distribution.

Proof. See Appendix A.

To establish the stability of the generated distributions, we first formally introduce the structural stability and Lipschitz continuity constants characterizing the underlying drift vector field components in (6). As stated in Section II-B, graph filters and GNNs are fundamentally stable operators. For a perturbation $\| \mathbf { E } ^ { \star } \| _ { 2 } ~ \leq ~ \varepsilon$ and its corresponding optimal permutation $\mathbf { P } _ { 0 } .$ , we define the structural stability constants $\Gamma _ { H }$ and $\Gamma _ { u }$ such that the filter and GNN in (6) satisfy

$$
\| \mathbf { P } _ { 0 } ^ { \top } \mathbf { H } _ { t } ( \widetilde { \mathbf { L } } ) \mathbf { x } - \mathbf { H } _ { t } ( \mathbf { L } ) \mathbf { P } _ { 0 } ^ { \top } \mathbf { x } \| \leq \big ( \Gamma _ { H } \varepsilon + \mathcal { O } ( \varepsilon ^ { 2 } ) \big ) \| \mathbf { x } \| ,\tag{8}
$$

$$
\| \mathbf { P } _ { 0 } ^ { \top } u _ { t } ^ { \theta } ( \mathbf { x } ; \widetilde { \mathbf { L } } ) - u _ { t } ^ { \theta } ( \mathbf { P } _ { 0 } ^ { \top } \mathbf { x } ; \mathbf { L } ) \| \leq \left( \Gamma _ { u } \varepsilon + \mathcal { O } ( \varepsilon ^ { 2 } ) \right) \| g ( \mathbf { x } , t ) \| ,\tag{9}
$$

for all $\textbf { x } \in \ \mathbb { R } ^ { N }$ and $t ~ \in ~ [ 0 , 1 ]$ . By Lipschitz continuity, both operators also satisfy a one-sided Lipschitz condition.

We denote their respective one-sided Lipschitz constants as $m _ { H }$ and $m _ { u } ,$ , satisfying

$$
\langle \mathbf { H } _ { t } ( \mathbf { L } ) ( \mathbf { x } - \mathbf { y } ) , \mathbf { x } - \mathbf { y } \rangle \leq m _ { H } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } ,\tag{10}
$$

$$
\begin{array} { r } { \langle u _ { t } ^ { \theta } ( \mathbf { x } ; \mathbf { L } ) - u _ { t } ^ { \theta } ( \mathbf { y } ; \mathbf { L } ) , \mathbf { x } - \mathbf { y } \rangle \leq m _ { u } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } , } \end{array}\tag{11}
$$

for all $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { N }$ and $t \in [ 0 , 1 ]$ , where $m _ { H }$ and $m _ { u }$ are chosen as uniform upper bounds for the nominal GSO L and the admissible perturbed GSOs Le. Combining these properties yields the structural stability and continuity of the full drift vector field $b _ { t } ( \mathbf { x } ; \mathbf { L } )$ , as asserted in the following lemmata.

Lemma 1. The vector field $b _ { t } ( \mathbf { x } ; \mathbf { L } )$ in (6) is structurally stable, i.e., for the optimal permutation $\mathbf { P } _ { 0 }$ it holds that

$$
\begin{array} { r } { \| \mathbf { P } _ { 0 } ^ { \top } b _ { t } ( \mathbf { x } ; \widetilde { \mathbf { L } } ) - b _ { t } ( \mathbf { P } _ { 0 } ^ { \top } \mathbf { x } ; \mathbf { L } ) \| \leq \varepsilon \big ( \Gamma _ { H } \| \mathbf { x } \| + \Gamma _ { u } \| g ( \mathbf { x } , t ) \| \big ) } \\ { + \mathcal { O } ( \varepsilon ^ { 2 } ) ( \| \mathbf { x } \| + \| g ( \mathbf { x } , t ) \| ) , } \end{array}
$$

for all $\mathbf { x } \in \mathbb { R } ^ { N }$ and $t \in [ 0 , 1 ]$

Proof. See Appendix B.

Lemma 2. The vector field $b _ { t } ( \mathbf { x } ; \mathbf { L } )$ in (6) is one-sided Lipschitz with combined constant $m _ { b } = m _ { H } + m _ { u } , i . e .$

$$
\begin{array} { r } { \langle b _ { t } ( \mathbf { x } ; \mathbf { L } ) - b _ { t } ( \mathbf { y } ; \mathbf { L } ) , \mathbf { x } - \mathbf { y } \rangle \leq m _ { b } \| \mathbf { x } - \mathbf { y } \| ^ { 2 } , } \end{array}
$$

for all $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { N }$ and $t \in [ 0 , 1 ]$ . The same bound holds for the perturbed GSO Le.

Proof. See Appendix C.

Equipped with these properties, we establish the following stability result for the distributions generated by the SDE (5).

Theorem 1. For any initial distribution $\mathbf { x } _ { 0 } \sim p _ { 0 } ,$ , assume that the components of the drift $b _ { t } ( \mathbf { x } ; \mathbf { L } )$ in (6) are permutation equivariant, and that $b _ { t }$ satisfies the structural stability and one-sided Lipschitz properties stated in Lemmata 1 and 2. Then, the Wasserstein-2 (W<sub>2</sub>) distance between the probability laws generated by the nominal and perturbed dynamics (cf. (7)) satisfies

$$
\operatorname* { m i n } _ { \mathbf { P } \in \mathcal { P } } W _ { 2 } \left( \widetilde { \pi } _ { t } ^ { \mathbf { P } } , \pi _ { t } ^ { \mathbf { P } } \right) \leq \Omega _ { t } \big ( \Gamma _ { H } C _ { H , t } + \Gamma _ { u } C _ { u , t } \big ) \varepsilon + O ( \varepsilon ^ { 2 } ) ,\tag{12}
$$

for all $t \in [ 0 , 1 ]$ , where $\begin{array} { r } { \Omega _ { t } : = \int _ { 0 } ^ { t } e ^ { m _ { b } ( t - s ) } d s } \end{array}$ and

$$
\begin{array} { r l } & { C _ { H , t } : = \underset { \mathbf { P } \in \mathcal { P } } { \operatorname* { m a x } } \underset { s \in [ 0 , t ] } { \operatorname* { s u p } } \left( \mathbb { E } \left[ \left\| \Phi _ { s } ( \mathbf { P } ^ { \top } \mathbf { x } _ { 0 } ; \mathbf { L } ) \right\| ^ { 2 } \right] \right) ^ { \frac { 1 } { 2 } } , } \\ & { C _ { u , t } : = \underset { \mathbf { P } \in \mathcal { P } } { \operatorname* { m a x } } \underset { s \in [ 0 , t ] } { \operatorname* { s u p } } \left( \mathbb { E } \left[ \left\| g \big ( \Phi _ { s } ( \mathbf { P } ^ { \top } \mathbf { x } _ { 0 } ; \mathbf { L } ) , s \big ) \right\| ^ { 2 } \right] \right) ^ { \frac { 1 } { 2 } } . } \end{array}
$$

Proof. See Appendix D.

We find that stability is governed by three ingredients: (i) $\Gamma _ { H }$ and $\Gamma _ { u } ,$ which capture the inherent stability of the graph filter and GNN adopted for the parameterization; (ii) $\Omega _ { t } .$ , which is exponential in the combined Lipschitz constant $m _ { b }$ and dictates how the continuous dynamics amplify errors; and (iii) $C _ { H , t }$ and $C _ { u , t }$ , the trajectory suprema across all initial condition permutations. Evaluating the bound in Theorem 1 at the terminal t = 1 yields the stability of generated samples.

## IV. STABILIZING GRAPH FILTERS

With an eye towards enhanced stability, the GNN constants $\{ \Gamma _ { u } , m _ { u } \}$ can be regularized during training [34, Sec. V] but the filter constants $\{ \Gamma _ { H } , m _ { H } \}$ are predetermined by the chosen reference dynamics. We now specialize the general reference dynamics to a class amenable to filter synthesis. Let the GSO L be a normalized graph Laplacian with spectrum contained in [0, 2]. We consider a degenerate zero-diffusion setting [17], with $\pmb { \alpha } _ { t } = \mathbf { 0 } , \pmb { \Sigma } _ { t } = \mathbf { 0 }$ , and a time-independent graph filter

$$
\mathbf { H } ( \mathbf { L } ) = \sum _ { k = 0 } ^ { K } h _ { k } \mathbf { L } ^ { k } .\tag{13}
$$

The reference dynamics in (4) therefore reduce to

$$
d \mathbf { y } _ { t } = \mathbf { H } ( \mathbf { L } ) \mathbf { y } _ { t } d t ,\tag{14}
$$

for some initial condition $\mathbf { y } _ { 0 }$

Example 1 (Graph Heat Diffusion). A canonical choice is ${ \bf H ( L ) } = - \kappa { \bf L } , \kappa > 0$ , which yields the graph heat equation

$$
d \mathbf { y } _ { t } = - \kappa \mathbf { L y } _ { t } d t .\tag{15}
$$

These dynamics diffuse the signal over G and progressively promote spatial smoothness, making heat diffusion a natural model for a broad class of network processes [47], [48].

Motivated by Theorem 1 and the success of the heat equation in graph signal generation [21]–[23], we aim to design a graph filter that preserves the smoothing properties of (15) while boosting stability against structural perturbations.

## A. Stability Constraints

Theorem 1 establishes that the filter constants $\{ \Gamma _ { H } , m _ { H } \}$ affect the stability of the generative dynamics. First, we characterize the one-sided Lipschitz constant $m _ { H }$

Proposition 2. Let $\mathbf { H } ( \mathbf { L } )$ be a graph filter with frequency response $h ( \lambda )$ , and let $\{ \lambda _ { i } \} _ { i = 1 } ^ { N }$ denote the eigenvalues of the GSO L. Then, the one-sided Lipschitz constant of H(L) is

$$
m _ { H } = \operatorname* { m a x } _ { i \in \{ 1 , . . . , N \} } h ( \lambda _ { i } ) .\tag{16}
$$

Proof. See Appendix E.

To rule out artificial stability improvements arising from uniform contraction effects (which act independently of the graph structure), we restrict our design space by setting $h _ { 0 } = 0$ in (13). Hence, the graph-frequency response takes the form

$$
h ( \lambda ) = \sum _ { k = 1 } ^ { K } h _ { k } \lambda ^ { k } ,\tag{17}
$$

parameterized by the coefficients h $\mathbf { \Psi } : = [ h _ { 1 } , \ldots , h _ { K } ] ^ { \top }$ . Because $h _ { 0 } = 0$ , we have $h ( 0 ) = 0$ . Moreover, $\lambda = 0$ is an eigenvalue of the graph Laplacian, so the smallest achievable one-sided Lipschitz constant is $m _ { H } ~ = ~ 0$ [cf. (16)]. For given graph G, it would suffice to enforce $h ( \lambda _ { i } ) \leq 0$ at its eigenvalues. To obtain a graph-uniform guarantee valid for any normalized graph Laplacian, we instead impose

$$
h ( \lambda ) \leq 0 , \qquad \forall \lambda \in [ 0 , 2 ] .\tag{18}
$$

This ensures $m _ { H } = 0$ regardless of the graph’s spectrum.

Second, we address the structural stability constant $\Gamma _ { H }$ . As discussed in Section II-B, Γ<sub>H</sub> is proportional to the filter’s integral Lipschitz constant $C _ { \mathrm { I L } } ( h )$ in $( 2 )$ . Having $m _ { H } = 0$ our remaining objective is to minimize $C _ { \mathrm { I L } } ( h ) ;$ see (20). Note that this criterion differs from (and refines) prior work [1], which minimized the surrogate bound $\operatorname* { m a x } _ { \lambda } \left| \lambda h ^ { \prime } ( \lambda ) \right|$ rather than optimizing the exact integral Lipschitz constant directly.

## B. Smoothness Constraints

To quantify the smoothness induced by the reference dynamics, we consider the Dirichlet energy of the terminal signal. The solution of (14) at $t \ = \ 1$ has graph-frequency representation

$$
\hat { y } _ { 1 , i } = e ^ { h ( \lambda _ { i } ) } \hat { y } _ { 0 , i } , \qquad i = 1 , \ldots , N .
$$

Its Dirichlet energy is therefore

$$
\mathbf { y } _ { 1 } ^ { \top } \mathbf { L } \mathbf { y } _ { 1 } = \sum _ { i = 1 } ^ { N } \lambda _ { i } e ^ { 2 h ( \lambda _ { i } ) } | \hat { y } _ { 0 , i } | ^ { 2 } .
$$

For $\| \mathbf { y } _ { 0 } \| \leq 1$ , the worst-case Dirichlet energy satisfies

$$
\mathbf { y } _ { 1 } ^ { \top } \mathbf { L } \mathbf { y } _ { 1 } \leq \operatorname* { m a x } _ { \lambda \in [ 0 , 2 ] } \lambda e ^ { 2 h ( \lambda ) }
$$

and we therefore impose the smoothness constraint

$$
\operatorname* { m a x } _ { \lambda \in [ 0 , 2 ] } \lambda e ^ { 2 h ( \lambda ) } \le \eta ,
$$

for a prescribed tolerance $\eta \in ( 0 , 2 )$ . Taking logarithms, this translates to an explicit requirement on the filter’s graph frequency response, namely

$$
h ( \lambda ) \leq { \frac { 1 } { 2 } } \log \left( { \frac { \eta } { \lambda } } \right) , \qquad { \mathrm { f o r ~ a l l ~ } } \lambda \in ( 0 , 2 ] .\tag{19}
$$

## C. Stabilizing Filter Design Problem

Combining the stability and smoothness desiderata, we seek filter coefficients h $\mathbf { \Psi } _ { \in \mathbb { R } ^ { K } }$ that minimize the integral Lipschitz constant $C _ { \mathrm { I L } } ( h )$ while satisfying the admissibility and smoothness constraints in (18) and (19), respectively. This gives the proposed stabilizing filter design problem

$$
\operatorname* { m i n } _ { \mathbf { h } } ~ C _ { \mathrm { I L } } ( h )
$$

$$
\mathrm { s . t . } \quad h ( \lambda ) \leq \operatorname* { m i n } \left\{ 0 , \frac { 1 } { 2 } \log \left( \frac { \eta } { \lambda } \right) \right\} , \qquad \lambda \in ( 0 , 2 ] .\tag{20}
$$

Using the definition of $C _ { \mathrm { I L } } ( h )$ in (2), we can equivalently write (20) in its epigraph form as

min C h,C

$$
\begin{array} { r l } { \mathrm { s . t . } } & { | h ( \lambda _ { 2 } ) - h ( \lambda _ { 1 } ) | \leq 2 C \displaystyle \frac { \lambda _ { 2 } - \lambda _ { 1 } } { \lambda _ { 1 } + \lambda _ { 2 } } , \quad 0 \leq \lambda _ { 1 } < \lambda _ { 2 } \leq 2 , } \\ & { h ( \lambda ) \leq \operatorname* { m i n } \left\{ 0 , \displaystyle \frac { 1 } { 2 } \log \left( \displaystyle \frac { \eta } { \lambda } \right) \right\} , \quad \lambda \in ( 0 , 2 ] . } \end{array}\tag{21}
$$

We denote the optimal value of (21) by $C _ { K } ^ { \star }$ . Since $h ( \lambda )$ in (17) is linear in the polynomial coefficients h, (21) is a semiinfinite linear program. In practice, it can be approximated by discretizing (0, 2] over a dense grid and solving the resulting finite-dimensional linear program [49, Sec. 7]. Representative numerically optimized filter responses for different polynomial orders are shown in Fig. 1.

![](images/840043793afd463f0a16dc59ae03710eb8bf829a9e4a970b8835fafd59725ccc.jpg)  
Fig. 1. Numerical solutions to (21) for $\eta = 0 . 1$ and different polynomial orders K, together with the smoothness bound.

## D. Optimal Value and Explicit Constructions

We will now characterize the convergence of (21) to a fundamental limit, as the filter order K grows. First, we observe that a lower bound on the optimal value $C _ { K } ^ { \star }$ can be shown to be

$$
C _ { K } ^ { \star } \geq \frac { a } { 2 } , \quad \mathrm { w h e r e ~ } a = \frac { 1 } { 2 } \ln \left( \frac { 2 } { \eta } \right) .\tag{22}
$$

This follows by evaluating the integral Lipschitz quotient at the endpoints $( \lambda _ { 1 } , \lambda _ { 2 } ) \ = \ ( 0 , 2 )$ ; see also Proposition 3 for further details. Notice that the bound is independent of K.

Consider now the degree-K family of filters with response

$$
h _ { K , b _ { K } } ( \lambda ) : = - b _ { K } \left[ 1 - \left( 1 - \frac { \lambda } { 2 } \right) ^ { K } \right] ,\tag{23}
$$

where

$$
b _ { K } : = \operatorname* { m a x } _ { x \in [ \eta / 2 , 1 ] } \frac { a + \frac { 1 } { 2 } \ln x } { 1 - ( 1 - x ) ^ { K } } .\tag{24}
$$

Proposition 3 asserts that these polynomials satisfy the constraints of (21) for all $K \ \geq \ 2 ,$ and that their integral Lipschitz constants approach the lower bound (22) at a rate of $\bar { \mathcal { O } ( { r } ^ { K } ) }$ , where $\begin{array} { r } { r = 1 - \frac { \eta } { 2 } } \end{array}$ . This provides an upper bound for the convergence of $C _ { K } ^ { \star }$ and an explicit polynomial construction that approaches the theoretical optimum at an exponential rate; representative responses are shown in Fig. 2.

Proposition 3. For every $K \geq 2 ,$ let $h _ { K , b _ { K } }$ be given by (23). Then $h _ { K , b _ { K } }$ is feasible for (21), and

$$
\frac { a } { 2 } \leq C _ { K } ^ { \star } \leq C _ { \mathrm { I L } } \big ( h _ { K , b _ { K } } \big ) \leq \frac { a } { 2 ( 1 - r ^ { K } ) } .
$$

Hence,

$$
C _ { K } ^ { \star } \longrightarrow \frac { a } { 2 } , \qquad C _ { K } ^ { \star } - \frac { a } { 2 } = \mathcal { O } ( r ^ { K } ) .
$$

Proof. See Appendix F.

Proposition 3 established that the optimal integral Lipschitz constant approaches the universal lower bound exponentially with the polynomial order. The lower bound, however, need

![](images/3dd124ea0e7339f862d29ed3ac01fb5553dc662befe044b3354eeecaeb7b6b4a.jpg)  
Fig. 2. Filters $h _ { K , b _ { K } }$ in (23) for $\eta = 0 . 1$ and different polynomial orders K, together with the smoothness bound. The claimed feasibility is apparent.

not only be approached asymptotically. A finite-order polynomial can attain it exactly, as the next result shows.

Proposition 4. Let $h _ { K , b _ { K } }$ be given by (23). If

$$
K \geq K _ { 0 } : = \operatorname* { m a x } \left\{ 2 , 1 + \left\lceil { \frac { \ln ( 2 a ) } { - \ln r } } \right\rceil \right\} ,
$$

then $b _ { K } = a ,$ , and

$$
C _ { \mathrm { I L } } \big ( h _ { K , b _ { K } } \big ) = C _ { K } ^ { \star } = \frac { a } { 2 } .
$$

Proof. See Appendix G.

The threshold $K _ { 0 }$ is only a sufficient condition and is generally conservative. In practice, the optimum may be attained at substantially smaller orders. The main significance of Proposition 4 is that the optimal value $a / 2$ can always be attained by a finite-order polynomial.

## V. NUMERICAL EXPERIMENTS

We test the proposed stabilizing filters in two settings: (i) a synthetic setting based on a Stochastic Block Model (SBM) graph [50]; and (ii) a real-world scenario using fMRI data [51], where functional brain connectivity defines the graph structure. In the synthetic test case, the objective is data generation, transporting signals from an initial Gaussian distribution to a target data distribution. In the fMRI setting, the objective is distribution matching, transporting signals from a source data distribution (liberal) to a target distribution (aligned).

For both settings, we consider two experiments. First, we investigate the structural stability of the reference dynamics

$$
d { \bf y } _ { t } = { \bf H } ( { \bf L } ) { \bf y } _ { t } d t , \qquad { \bf y } _ { 0 } \sim p _ { 0 } ,\tag{25}
$$

whose solution at time t is denoted by $\Psi _ { t } ( \mathbf { y } _ { 0 } ; \mathbf { L } )$ , comparing the numerical solution of (21) with the analytical construction in (23). Second, we evaluate the full generative dynamics

$$
d \mathbf { x } _ { t } = \left[ \mathbf { H } ( \mathbf { L } ) \mathbf { x } _ { t } + u _ { t } ^ { \pmb { \theta } } ( \mathbf { x } _ { t } ; \mathbf { L } ) \right] d t , \qquad \mathbf { x } _ { 0 } \sim p _ { 0 } ,\tag{26}
$$

whose solution at time t is denoted by $\Phi _ { t } ( \mathbf { x } _ { 0 } ; \mathbf { L } )$ , assessing both generative performance and structural stability under perturbations of the underlying graph with nominal GSO L.

## A. Experimental Setup

Datasets. In the synthetic setting, we use an SBM graph [50] with two communities of 10 nodes each $( N \ = \ 2 0 )$ The intra- and inter-community edge formation probabilities are 0.8 and 0.1, respectively. Signals in each community are drawn from a Gaussian distribution with standard deviation 1. One community has mean 1, while the other has mean −1. We generate 150 signals for training and validation (80% and 20%, respectively) and an additional 150 signals for testing.

For the fMRI experiments, we consider a single subject from the HCP dataset [51]. Data live in $\mathbb { R } ^ { 3 6 0 \times 1 \bar { 1 } 9 0 }$ , where $N = 3 6 0$ is the number of brain regions (nodes) and 1190 is the length of temporal window. Each time point is treated as an individual graph signal $\mathbf { x } ^ { ( i ) } \in \mathbb { R } ^ { 3 6 0 }$ , yielding a dataset $\{ \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( 1 1 9 0 ) } \}$ . To construct the source and target distributions for the distribution matching task, we decompose the signals into high and low frequency components using graph spectral filtering. Specifically, we use the 30th eigenvalue of the original graph normalized Laplacian as the threshold, applying a high-pass filter to obtain the liberal signals (initial distribution) and a low-pass filter to obtain the aligned signals (final distribution). These signals remain fixed throughout the experiments. We split the dataset into 70% for training, 10% for validation, and 20% for testing, constructing the nominal GSO L solely from training signals.

Graph perturbations and evaluation. Graph perturbations are introduced exclusively during testing to evaluate structural stability, while training is performed using the original graphs. For the SBM graphs, we introduce a controlled synthetic relative perturbation of the form [cf. (1)]

$$
\widetilde { \mathbf { L } } = \mathbf { L } + \frac { 1 } { 2 } ( \mathbf { E } \mathbf { L } + \mathbf { L } \mathbf { E } ) , \quad \mathrm { w i t h ~ } \| \mathbf { E } \| _ { 2 } = \varepsilon .
$$

Here, E is obtained by sampling a random symmetric matrix M and rescaling it as $\begin{array} { r c l } { \mathbf { E } } & { = } & { \varepsilon \mathbf { M } / \| \mathbf { M } \| _ { 2 } } \end{array}$ , ensuring $\| \mathbf { E } \| _ { 2 } ~ = ~ \varepsilon .$ For the fMRI data, graph perturbations are obtained by computing empirical correlation matrices from different-sized subsets of the training data. We quantify structural stability by the variation in terminal outputs, namely, $\| \Psi _ { 1 } ( \mathbf { y } _ { 0 } ; \widetilde { \mathbf { L } } ) - \Psi _ { 1 } ( \mathbf { y } _ { 0 } ; \mathbf { L } ) \|$ for the reference dynamics and $\| \Phi _ { 1 } ( \mathbf { x } _ { 0 } ; \widetilde { \mathbf { L } } ) - \Phi _ { 1 } ( \mathbf { x } _ { 0 } ; \mathbf { L } ) \|$ for the complete generative model. For the latter, we additionally evaluate generative performance using the Wasserstein-1 $( W _ { 1 } )$ distance between the generated and target distributions. Trajectories are approximated using an Euler sampler with 100 steps.

Architecture. We parameterize the vector field $u _ { t } ^ { \theta } ( \mathbf { x } _ { t } ; \mathbf { L } )$ using a three-layer GNN $( L = 3 )$ . The temporal conditioning function is defined as $g ( \mathbf { x } , t ) ~ = ~ [ \mathbf { x } , \mathbf { 1 e m b } ( t ) ^ { \top } ]$ , where emb $( t ) \in \mathbb { R } ^ { 1 2 8 }$ is a standard 128-dimensional sinusoidal time embedding. For the synthetic SBM case, we use three hidden layers with widths [64, 32, 16], whereas for the fMRI case, we use [256, 64, 32]. In both cases, the input consists of a 1-dimensional node feature and a 128-dimensional time embedding, resulting in an input feature dimension of $F _ { 0 } = 1 2 9$ The output layer has a single feature, $F _ { \mathrm { o u t } } = 1$ . Each graph convolutional layer utilizes $P + 1 = 4$ filter taps and use the normalized graph Laplacian as the GSO. We employ SiLU as the pointwise nonlinearity $\sigma ( \cdot )$ . The smoothing tolerance is set to $\eta = 0 . 0 4$ for SBM and $\eta = 0 . 1$ for fMRI, based on preliminary experimentation.

![](images/1ef97635ae912cc681bbcdef62d7892ea2630fed65184fd04daeee7d0b94e859.jpg)

![](images/32530e09c5a85daf755448b43642be9e8f8e56728713d9cd79da124a51fd4be1.jpg)  
Fig. 3. Structural stability of the deterministic reference dynamics (25). (Left) SBM graph under controlled synthetic perturbations of magnitude $\varepsilon ,$ and (right) fMRI graph constructed using different training set ratios. Solid lines correspond to the numerically optimized filters obtained from $( 2 1 )$ , while dotted lines represent the analytical construction in (23), for $\eta = 0 . 1$ . In both settings, structural stability improves with increasing polynomial order $K ,$ , with the analytical filters exhibiting behavior comparable to their numerical counterparts.

Algorithm 1 Topological Flow Matching   
Require: $\mathbf { L } \ = \ \mathbf { V } \pmb { \Lambda } \mathbf { V } ^ { \top }$ , polynomial order $K ,$ smoothness   
tolerance $\eta ,$ coupling $\gamma ,$ learning rate $\rho ,$ network $u _ { t } ^ { \theta }$   
1: Obtain $\{ h _ { k } \} _ { k = 0 } ^ { K }$ numerically via (21) or analytically via   
(23)–(24)   
2: Compute $\begin{array} { r } { h _ { i } = \sum _ { k = 0 } ^ { K } h _ { k } \lambda _ { i } ^ { k } , i = 1 , \dots , N } \end{array}$   
3: while not converged do   
4: Sample $( { \bf y } _ { 0 } , { \bf y } _ { \perp } ) \sim \gamma , t \sim \mathcal { U } [ 0 , 1 ]$   
5: GFT: $\hat { \mathbf { y } } _ { 0 } = \mathbf { V } ^ { \top } \mathbf { y } _ { 0 } , \hat { \mathbf { y } } _ { 1 } = \mathbf { V } ^ { \top } \mathbf { y } _ { 1 }$   
6: Compute conditional path $\hat { \mathbf { y } } _ { t }$ via (31)   
7: Compute conditional vector field $\hat { u } _ { t } ^ { \mathbf { y } _ { 0 } , \mathbf { y } _ { 1 } }$ via (32)   
8: Inverse GFT: $\mathbf { y } _ { t } = \mathbf { V } \hat { \mathbf { y } } _ { t } , u _ { t } ^ { \mathbf { y } _ { 0 } , \mathbf { y } _ { 1 } } = \mathbf { \bar { V } } \hat { u } _ { t } ^ { \mathbf { y } _ { 0 } , \mathbf { y } _ { 1 } }$   
9: $\pmb { \theta } \gets \pmb { \theta } - \rho \nabla _ { \pmb { \theta } } \left\| u _ { t } ^ { \pmb { \theta } } ( \mathbf { y } _ { t } ; \mathbf { L } ) - u _ { t } ^ { \mathbf { y } _ { 0 } , \mathbf { y } _ { 1 } } \right\| ^ { 2 }$   
10: end while

Training. We train the learned vector field $u _ { t } ^ { \theta } ( \mathbf { x } _ { t } ; \mathbf { L } )$ using topological flow matching [17], regressing against graph-aware conditional vector fields with an independent coupling as detailed in Algorithm 1. We further incorporate a structural stability regularization term into the training objective to control the Lipschitz constant of the GNN, following [34, Sec. V]. For completeness, the corresponding conditional paths and vector fields are derived in Appendix H.

## B. Results and Discussion

Reference dynamics. Figure 3 illustrates the structural stability of the isolated reference filters. As the polynomial order K increases, the deviation between the terminal signals decreases, demonstrating improved robustness against graph perturbations. This trend holds for both the synthetic SBM graph (left) and the real-world fMRI (right). Also, the analytical filter construction in (23) closely matches the stability performance of the numerically optimized filters in (21).

Generative model. Figure 4 evaluates the full generative dynamics. While the learned GNN vector field $u _ { t } ^ { \theta }$ introduces an additional source of error, we still observe stability gains induced by the optimized graph filters. Specifically, higher-order filters $( K = 2 , 4 )$ exhibit lower output variation under perturbations compared to the heat equation baseline $( K = 1 )$ across both datasets (right column). Importantly, the Wasserstein-1 distance metrics (left column) confirm that these structural stability improvements are achieved without degrading the performance of the generative model. Unlike the isolated reference dynamics, the stability of the complete generative model also depends on the learned GNN vector field, whose structural stability may vary across training runs. This additional variability makes the stability trends less pronounced, although the stabilizing effect of the reference filters generally carries over to the complete generative dynamics.

## VI. CONCLUSION

This paper established a principled framework for designing stabilizing graph filters for graph-aware generative models. We demonstrate both theoretically and empirically that judiciously designed higher-order filters can improve structural stability over classical graph heat diffusion, while preserving generative performance. The novel graph filters can be obtained either by solving the semi-infinite linear program in (21) for an optimal solution, or through the explicit polynomial construction in (23). For the latter, the parameter $b _ { K }$ can be optimized for a prescribed smoothness level as in (24), or else treated as a free design parameter.

An important consideration is that the proposed design is fundamentally tied to the relative perturbation model in (1), which is well suited to graph-structured data [43, Sec. IV-$\mathbf { A l }$ . Different perturbation models lead to different stability bounds and, consequently, different optimal filter designs. For instance, under an absolute perturbation model, stability of GNNs is governed by the standard Lipschitz constant rather than the integral Lipschitz constant [29, Thm. 1], in which case the classical heat equation would be optimal.

Finally, although stability bounds may be numerically loose in practice [1], [29], [34], their main value lies in their ability to expose the factors governing structural sensitivity. In this sense, they provide a principled foundation for both, the robust filter design developed in this work, and the regularization of the learned GNN vector field in [34].

![](images/ec4979a0df0b47c3f1d3f629a8baeaff4a7ec35c7f970f4b028312fff7f73867.jpg)

![](images/413629fe06bc762e3fa843df75ddcb280976a954dfcaca4008c88238cd3beed1.jpg)

![](images/6698ee818007edc58341fa449d46def1a9caadeffeaab098420b07185288fdfe.jpg)

![](images/b1c22252f44eab692138e39bf617c3943f0eb106e011ec750187c17db854b362.jpg)  
Fig. 4. Generative performance and structural stability of the generative dynamics (26) under graph perturbations. (Top) SBM graph under controlled synthetic perturbations of magnitude ε, and (bottom) fMRI graphs constructed using different training set ratios. (Left) Generative performance measured by the Wasserstein-1 (W ) distance shows that higher-order optimized filters $( K = \overline { { 2 } } , 4 )$ achieve comparable or better performance than the heat equation baseline (K = 1). (Right) Empirical stability measured by the variation in generated outputs under graph perturbations improves with increasing polynomial order K. Lines and shaded regions represent medians and 25th–75th percentiles across 10 independent runs.

## APPENDIX A PROOF OF PROPOSITION 1

Proof. Let $\mathbf { z } _ { t } : = \mathbf { P } ^ { \top } \mathbf { x } _ { t }$ . Then

$$
\begin{array} { r } { d \mathbf { z } _ { t } = \mathbf { P } ^ { \top } d \mathbf { x } _ { t } = \mathbf { P } ^ { \top } b _ { t } ( \mathbf { x } _ { t } ; \mathbf { L } ) d t + \mathbf { P } ^ { \top } \Sigma _ { t } d \mathbf { w } _ { t } . } \end{array}
$$

Since $b _ { t } ( \mathbf { x } _ { t } ; \mathbf { L } )$ in (6) is a sum of permutation-equivariant terms, we have

$$
\mathbf { P } ^ { \top } b _ { t } ( \mathbf { x } _ { t } ; \mathbf { L } ) = b _ { t } ( \mathbf { P } ^ { \top } \mathbf { x } _ { t } ; \mathbf { P } ^ { \top } \mathbf { L } \mathbf { P } ) = b _ { t } ( \mathbf { z } _ { t } ; \mathbf { P } ^ { \top } \mathbf { L } \mathbf { P } ) .
$$

Let $d \widetilde { \mathbf { w } } _ { t } : = \mathbf { P } ^ { \top } d \mathbf { w } _ { t }$ . Since P is orthogonal, $\widetilde { \mathbf { w } } _ { t }$ is also a standard Brownian motion. Moreover, since $\mathbf { P } ^ { \top } \pmb { \Sigma } _ { t } \mathbf { P } = \pmb { \Sigma } _ { t } .$ we have

$$
\mathbf { P } ^ { \top } \Sigma _ { t } d \mathbf { w } _ { t } = \Sigma _ { t } \mathbf { P } ^ { \top } d \mathbf { w } _ { t } = \Sigma _ { t } d \widetilde { \mathbf { w } } _ { t } .
$$

Therefore,

$$
\begin{array} { r } { d \mathbf { z } _ { t } = b _ { t } ( \mathbf { z } _ { t } ; \mathbf { P } ^ { \top } \mathbf { L } \mathbf { P } ) d t + \Sigma _ { t } d \widetilde { \mathbf { w } } _ { t } , \qquad \mathbf { z } _ { 0 } = \mathbf { P } ^ { \top } \mathbf { x } _ { 0 } . } \end{array}
$$

Thus, $\mathbf { z } _ { t }$ satisfies the SDE associated with the initial condition $\mathbf { P } ^ { \top } \mathbf { x } _ { 0 }$ and graph $\mathbf { P } ^ { \top } \mathbf { L P }$ , driven by the standard Brownian motion $\widetilde { \mathbf { w } } _ { t }$ . Since $\widetilde { \mathbf { w } } _ { t }$ and $\mathbf { w } _ { t }$ have the same probability law, uniqueness in law of the SDE implies

$$
\begin{array} { r } { \mathbf { P } ^ { \top } \Phi _ { t } ( \mathbf { x } _ { 0 } ; \mathbf { L } ) \stackrel { d } { = } \Phi _ { t } ( \mathbf { P } ^ { \top } \mathbf { x } _ { 0 } ; \mathbf { P } ^ { \top } \mathbf { L } \mathbf { P } ) , } \end{array}
$$

which proves the result.

## APPENDIX B PROOF OF LEMMA 1

Proof. From the definition of the vector field $b _ { t } ( \mathbf { x } ; \mathbf { L } )$ in (6) and leveraging the stability bounds for the graph filter (8) and the GNN (9), a triangle inequality argument yields

$$
\begin{array} { r l } & { \| \mathbf { P } _ { 0 } ^ { \top } b _ { t } ( \mathbf { x } ; \widetilde { \mathbf { L } } ) - b _ { t } ( \mathbf { P } _ { 0 } ^ { \top } \mathbf { x } ; \mathbf { L } ) \| } \\ & { \leq \| \mathbf { P } _ { 0 } ^ { \top } \mathbf { H } _ { t } ( \widetilde { \mathbf { L } } ) \mathbf { x } - \mathbf { H } _ { t } ( \mathbf { L } ) \mathbf { P } _ { 0 } ^ { \top } \mathbf { x } \| } \\ & { + \| \mathbf { P } _ { 0 } ^ { \top } u _ { t } ^ { \theta } ( \mathbf { x } ; \widetilde { \mathbf { L } } ) - u _ { t } ^ { \theta } ( \mathbf { P } _ { 0 } ^ { \top } \mathbf { x } ; \mathbf { L } ) \| } \\ & { \leq \varepsilon \big ( \Gamma _ { H } \| \mathbf { x } \| + \Gamma _ { u } \| g ( \mathbf { x } , t ) \| \big ) + \mathcal { O } ( \varepsilon ^ { 2 } ) \big ( \| \mathbf { x } \| + \| g ( \mathbf { x } , t ) \| \big ) . } \end{array}
$$

## APPENDIX C PROOF OF LEMMA 2

Proof. From the definition of the vector field $b _ { t } ( \mathbf { x } ; \mathbf { L } )$ in (6) and recalling the one-sided Lipschitz constants for the graph filter (10) and the GNN (11), we have

$$
\begin{array} { r l } & { \langle b _ { t } ( { \bf x } ; { \bf L } ) - b _ { t } ( { \bf y } ; { \bf L } ) , { \bf x } - { \bf y } \rangle } \\ & { \ = \langle { \bf H } _ { t } ( { \bf L } ) ( { \bf x } - { \bf y } ) , { \bf x } - { \bf y } \rangle + \langle u _ { t } ^ { \theta } ( { \bf x } ; { \bf L } ) - u _ { t } ^ { \theta } ( { \bf y } ; { \bf L } ) , { \bf x } - { \bf y } \rangle } \\ & { \ \le ( m _ { H } + m _ { u } ) \| { \bf x } - { \bf y } \| ^ { 2 } . } \end{array}
$$

## APPENDIX D PROOF OF THEOREM 1

Proof. Fix the optimal permutation $\mathbf { P } _ { 0 } \in \mathcal { P }$ associated with the perturbation model in (1). Let $\zeta \sim p _ { 0 }$ and let $\mathbf { w } _ { t }$ be a standard Brownian motion. Define the nominal process $\mathbf { x } _ { t }$ by

$$
d \mathbf { x } _ { t } = b _ { t } ( \mathbf { x } _ { t } ; \mathbf { L } ) d t + \Sigma _ { t } d \mathbf { w } _ { t } , \qquad \mathbf { x } _ { 0 } = \mathbf { P } _ { 0 } ^ { \top } \boldsymbol { \zeta } .
$$

Likewise, define the perturbed process $\widetilde { \mathbf { x } } _ { t }$ by

$$
d \widetilde { \bf x } _ { t } = b _ { t } ( \widetilde { \bf x } _ { t } ; \widetilde { \bf L } ) d t + \pmb { \Sigma } _ { t } d \widetilde { \bf w } _ { t } , \qquad \widetilde { \bf x } _ { 0 } = \zeta ,
$$

where $\widetilde { \mathbf { w } } _ { t } : = \mathbf { P } _ { 0 } \mathbf { w } _ { t }$ . Since $\mathbf { P } _ { 0 }$ is orthogonal, $\widetilde { \mathbf { w } } _ { t }$ is also a standard Brownian motion. Define $\mathbf { y } _ { t } : = \mathbf { P } _ { 0 } ^ { \top } \widetilde { \mathbf { x } } _ { t }$ and notice that by construction,

$$
\mathbf { x } _ { t } \sim \pi _ { t } ^ { \mathbf { P } _ { 0 } } , \qquad \mathbf { y } _ { t } \sim \widetilde { \pi } _ { t } ^ { \mathbf { P } _ { 0 } } .
$$

Moreover,

$$
\begin{array} { r l } & { d { \mathbf { y } } _ { t } = { \mathbf { P } } _ { 0 } ^ { \top } d \widetilde { \mathbf { x } } _ { t } } \\ & { \qquad = { \mathbf { P } } _ { 0 } ^ { \top } b _ { t } ( \widetilde { \mathbf { x } } _ { t } ; \widetilde { \mathbf { L } } ) d t + { \mathbf { P } } _ { 0 } ^ { \top } { \boldsymbol { \Sigma } } _ { t } d \widetilde { \mathbf { w } } _ { t } } \\ & { \qquad = { \mathbf { P } } _ { 0 } ^ { \top } b _ { t } ( { \mathbf { P } } _ { 0 } { \mathbf { y } } _ { t } ; \widetilde { \mathbf { L } } ) d t + { \mathbf { P } } _ { 0 } ^ { \top } { \boldsymbol { \Sigma } } _ { t } { \mathbf { P } } _ { 0 } d \mathbf { w } _ { t } . } \end{array}
$$

Using ${ \bf P } _ { 0 } ^ { \top } { \bf \Sigma } { \bf { \Sigma } } \Sigma _ { t } { \bf P } _ { 0 } = \Sigma _ { t }$ , we obtain

$$
d \mathbf { y } _ { t } = \mathbf { P } _ { 0 } ^ { \top } b _ { t } \big ( \mathbf { P } _ { 0 } \mathbf { y } _ { t } ; \widetilde { \mathbf { L } } \big ) d t + \boldsymbol { \Sigma } _ { t } d \mathbf { w } _ { t } , \qquad \mathbf { y } _ { 0 } = \mathbf { P } _ { 0 } ^ { \top } \boldsymbol { \zeta } = \mathbf { x } _ { 0 } .
$$

Next, define ${ \bf z } _ { t } : = { \bf y } _ { t } - { \bf x } _ { t }$ . Since the two processes have the same initial condition and are driven by the same Brownian motion, the diffusion terms cancel, and therefore

$$
d \mathbf { z } _ { t } = \left[ \mathbf { P } _ { 0 } ^ { \top } b _ { t } ( \widetilde { \mathbf { x } } _ { t } ; \widetilde { \mathbf { L } } ) - b _ { t } ( \mathbf { x } _ { t } ; \mathbf { L } ) \right] d t , \qquad \mathbf { z } _ { 0 } = \mathbf { 0 } .
$$

We have

$$
\begin{array} { l } { \displaystyle \frac { 1 } { 2 } \frac { d } { d t } \| { \mathbf z } _ { t } \| ^ { 2 } = \displaystyle \frac { 1 } { 2 } \frac { d } { d t } \langle { \mathbf z } _ { t } , { \mathbf z } _ { t } \rangle } \\ { \displaystyle = \Big \langle { \mathbf z } _ { t } , \frac { d } { d t } { \mathbf z } _ { t } \Big \rangle } \\ { \displaystyle = \Big \langle { \mathbf z } _ { t } , { \mathbf P } _ { 0 } ^ { \top } b _ { t } \big ( \widetilde { { \mathbf x } } _ { t } ; \widetilde { { \mathbf L } } \big ) - b _ { t } \big ( { \mathbf x } _ { t } ; { \mathbf L } \big ) \Big \rangle . } \end{array}
$$

Adding and subtracting $\mathbf { P } _ { 0 } ^ { \top } b _ { t } ( \mathbf { P } _ { 0 } \mathbf { x } _ { t } ; \widetilde { \mathbf { L } } )$ gives

$$
\begin{array} { r } { \displaystyle \frac { 1 } { 2 } \frac { d } { d t } \| \mathbf { z } _ { t } \| ^ { 2 } = \Big \langle \mathbf { z } _ { t } , \mathbf { P } _ { 0 } ^ { \top } \left( b _ { t } \big ( \widetilde { \mathbf { x } } _ { t } ; \widetilde { \mathbf { L } } \big ) - b _ { t } \big ( \mathbf { P } _ { 0 } \mathbf { x } _ { t } ; \widetilde { \mathbf { L } } \big ) \right) \Big \rangle } \\ { + \Big \langle \mathbf { z } _ { t } , \mathbf { P } _ { 0 } ^ { \top } b _ { t } \big ( \mathbf { P } _ { 0 } \mathbf { x } _ { t } ; \widetilde { \mathbf { L } } \big ) - b _ { t } \big ( \mathbf { x } _ { t } ; \mathbf { L } \big ) \Big \rangle . } \end{array}
$$

We bound the two terms separately. For the first term, using the orthogonality of $\mathbf { P } _ { 0 }$ and Lemma 2, we have

$$
\begin{array} { r l } & { \Bigl \langle \mathbf { z } _ { t } , \mathbf { P } _ { 0 } ^ { \top } \left( b _ { t } ( \widetilde { \mathbf { x } } _ { t } ; \widetilde { \mathbf { L } } ) - b _ { t } ( \mathbf { P } _ { 0 } \mathbf { x } _ { t } ; \widetilde { \mathbf { L } } ) \right) \Bigl \rangle } \\ & { \qquad = \Bigl \langle \widetilde { \mathbf { x } } _ { t } - \mathbf { P } _ { 0 } \mathbf { x } _ { t } , b _ { t } ( \widetilde { \mathbf { x } } _ { t } ; \widetilde { \mathbf { L } } ) - b _ { t } ( \mathbf { P } _ { 0 } \mathbf { x } _ { t } ; \widetilde { \mathbf { L } } ) \Bigl \rangle . } \\ & { \qquad \le m _ { b } \| \widetilde { \mathbf { x } } _ { t } - \mathbf { P } _ { 0 } \mathbf { x } _ { t } \| ^ { 2 } = m _ { b } \| \mathbf { z } _ { t } \| ^ { 2 } . } \end{array}
$$

For the second term, the Cauchy–Schwarz inequality gives

$$
\begin{array} { r l } & { \Bigl \langle \mathbf { z } _ { t } , \mathbf { P } _ { 0 } ^ { \top } b _ { t } ( \mathbf { P } _ { 0 } \mathbf { x } _ { t } ; \widetilde { \mathbf { L } } ) - b _ { t } ( \mathbf { x } _ { t } ; \mathbf { L } ) \Bigr \rangle } \\ & { \qquad \leq \| \mathbf { z } _ { t } \| \left\| \mathbf { P } _ { 0 } ^ { \top } b _ { t } ( \mathbf { P } _ { 0 } \mathbf { x } _ { t } ; \widetilde { \mathbf { L } } ) - b _ { t } ( \mathbf { x } _ { t } ; \mathbf { L } ) \right\| . } \end{array}
$$

Applying Lemma 1 with $\mathbf { P } _ { 0 } \mathbf { x } _ { t }$ as the input yields

$$
\begin{array} { r l } & { \left\| \mathbf { P } _ { 0 } ^ { \top } b _ { t } ( \mathbf { P } _ { 0 } \mathbf { x } _ { t } ; \widetilde { \mathbf { L } } ) - b _ { t } ( \mathbf { x } _ { t } ; \mathbf { L } ) \right\| } \\ & { \qquad \leq \varepsilon \left( \Gamma _ { H } \| \mathbf { P } _ { 0 } \mathbf { x } _ { t } \| + \Gamma _ { u } \| g ( \mathbf { P } _ { 0 } \mathbf { x } _ { t } , t ) \| \right) } \\ & { \qquad + \mathcal { O } ( \varepsilon ^ { 2 } ) \left( \| \mathbf { P } _ { 0 } \mathbf { x } _ { t } \| + \| g ( \mathbf { P } _ { 0 } \mathbf { x } _ { t } , t ) \| \right) . } \end{array}
$$

Since $\mathbf { P } _ { 0 }$ is orthogonal and g is permutation equivariant, then

$$
\begin{array} { r l } & { \Big \langle \mathbf { z } _ { t } , \mathbf { P } _ { 0 } ^ { \top } b _ { t } ( \mathbf { P } _ { 0 } \mathbf { x } _ { t } ; \widetilde { \mathbf { L } } ) - b _ { t } ( \mathbf { x } _ { t } ; \mathbf { L } ) \Big \rangle } \\ & { \qquad \leq \left( \varepsilon \left( \Gamma _ { H } \Vert \mathbf { x } _ { t } \Vert + \Gamma _ { u } \Vert g ( \mathbf { x } _ { t } , t ) \Vert \right) \right. } \\ & { \qquad \left. + \mathcal { O } ( \varepsilon ^ { 2 } ) \left( \Vert \mathbf { x } _ { t } \Vert + \Vert g ( \mathbf { x } _ { t } , t ) \Vert \right) \right) \Vert \mathbf { z } _ { t } \Vert . } \end{array}
$$

Combining the two bounds, we obtain

$$
\begin{array} { r } { \displaystyle \frac { 1 } { 2 } \frac { d } { d t } \| { \bf z } _ { t } \| ^ { 2 } \leq m _ { b } \| { \bf z } _ { t } \| ^ { 2 } + \Big ( \varepsilon \big ( \Gamma _ { H } \| { \bf x } _ { t } \| + \Gamma _ { u } \| g ( { \bf x } _ { t } , t ) \| \big ) } \\ { \displaystyle + \mathcal { O } ( \varepsilon ^ { 2 } ) \big ( \| { \bf x } _ { t } \| + \| g ( { \bf x } _ { t } , t ) \| \big ) \Big ) \| { \bf z } _ { t } \| . } \end{array}
$$

Let $r _ { t } : = \| \mathbf { z } _ { t } \|$ , and by the chain rule one has $\begin{array} { r } { \frac { 1 } { 2 } \frac { d } { d t } r _ { t } ^ { 2 } = r _ { t } \frac { d } { d t } r _ { t } } \end{array}$ For t such that $r _ { t } > 0$ , dividing by $r _ { t }$ gives

$$
\begin{array} { r } { \displaystyle \frac { d } { d t } r _ { t } \leq m _ { b } r _ { t } + \varepsilon \big ( \Gamma _ { H } \| \mathbf { x } _ { t } \| + \Gamma _ { u } \| g ( \mathbf { x } _ { t } , t ) \| \big ) } \\ { \displaystyle + \mathcal { O } ( \varepsilon ^ { 2 } ) \big ( \| \mathbf { x } _ { t } \| + \| g ( \mathbf { x } _ { t } , t ) \| \big ) . } \end{array}
$$

Let a denote the left endpoint of the connected component of $\lbrace s : \ r _ { s } \ > \ 0 \rbrace$ containing t. Since $r _ { 0 } ~ = ~ 0 ,$ , such a left endpoint exists, and by continuity $r _ { a } ~ = ~ 0$ . Applying Gronwall’s inequality¨ <sup>1</sup> on $[ a , t ]$ , and noting that $[ a , t ] \subseteq [ 0 , t ]$ and the resulting integrand is nonnegative, we obtain

$$
\begin{array} { r l } & { r _ { t } \leq \displaystyle \int _ { 0 } ^ { t } e ^ { m _ { b } ( t - s ) } \Big [ \varepsilon \big ( \Gamma _ { H } \| \mathbf { x } _ { s } \| + \Gamma _ { u } \| g ( \mathbf { x } _ { s } , s ) \| \big ) } \\ & { \qquad + \mathcal { O } ( \varepsilon ^ { 2 } ) \big ( \| \mathbf { x } _ { s } \| + \| g ( \mathbf { x } _ { s } , s ) \| \big ) \Big ] d s . } \end{array}
$$

The 2-Wasserstein distance is bounded by the cost of this coupling, namely $W _ { 2 } ( \widetilde { \pi } _ { t } ^ { \mathbf { P } _ { 0 } } , \pi _ { t } ^ { \mathbf { P } _ { 0 } } ) \leq \left( \mathbb { E } \big [ \| \mathbf { z } _ { t } \| ^ { 2 } \big ] \right) ^ { \frac { 1 } { 2 } }$ . Applying Minkowski’s integral inequality <sup>2</sup> and the $L ^ { 2 }$ triangle inequality $^ 3 ,$ , yields

$$
\begin{array} { l } { { \displaystyle \left( \mathbb { E } \left[ \| { \bf z } _ { t } \| ^ { 2 } \right] \right) ^ { \frac { 1 } { 2 } } \leq } } \\ { { \displaystyle \varepsilon \int _ { 0 } ^ { t } e ^ { m _ { b } ( t - s ) } \left( \Gamma _ { H } \left( \mathbb { E } \left[ \| { \bf x } _ { s } \| ^ { 2 } \right] \right) ^ { \frac { 1 } { 2 } } + \Gamma _ { u } \left( \mathbb { E } \left[ \| g ( { \bf x } _ { s } , s ) \| ^ { 2 } \right] \right) ^ { \frac { 1 } { 2 } } \right) d s } } \\ { { \displaystyle + \mathcal { O } ( \varepsilon ^ { 2 } ) \int _ { 0 } ^ { t } e ^ { m _ { b } ( t - s ) } \bigg ( \left( \mathbb { E } \left[ \| { \bf x } _ { s } \| ^ { 2 } \right] \right) ^ { \frac { 1 } { 2 } } + \left( \mathbb { E } \left[ \| g ( { \bf x } _ { s } , s ) \| ^ { 2 } \right] \right) ^ { \frac { 1 } { 2 } } \bigg ) d s . } } \end{array}
$$

Using $C _ { H , t } \geq \big ( \mathbb { E } [ \| \mathbf { x } _ { s } \| ^ { 2 } ] \big ) ^ { \frac { 1 } { 2 } }$ and $C _ { u , t } \ \geq \ \big ( \mathbb { E } [ \| g ( \mathbf { x } _ { s } , s ) \| ^ { 2 } ] \big ) ^ { \frac { 1 } { 2 } }$ yields

$$
\begin{array} { r l } & { \underset { { \bf { P } } \in { \mathcal { P } } } { \operatorname* { m i n } } W _ { 2 } \left( \widetilde { \pi } _ { t } ^ { { \bf { P } } } , \pi _ { t } ^ { { \bf { P } } } \right) \le W _ { 2 } \left( \widetilde { \pi } _ { t } ^ { { \bf { P } } _ { 0 } } , \pi _ { t } ^ { { \bf { P } } _ { 0 } } \right) } \\ & { \qquad \le \Omega _ { t } \bigl ( \Gamma _ { H } C _ { H , t } + \Gamma _ { u } C _ { u , t } \bigr ) \varepsilon + \mathcal { O } ( \varepsilon ^ { 2 } ) } \end{array}
$$

which is the desired structural stability bound.

$$
\begin{array} { r l } & { ^ { 1 } \mathrm { I f } \ r ^ { \prime } ( t ) \leq \mu r ( t ) + f ( t ) \ \mathrm { w i t h } \ r ( a ) = 0 , \mathrm { t h e n } \ r ( t ) \leq \int _ { a } ^ { t } e ^ { \mu ( t - s ) } f ( s ) d s . } \\ & { ^ { 2 } \bigg ( \mathbb { E } \left[ \left| \int _ { 0 } ^ { t } f _ { s } \ d s \right| ^ { 2 } \right] \bigg ) ^ { \frac { 1 } { 2 } } \leq \int _ { 0 } ^ { t } \left( \mathbb { E } \left[ | f _ { s } | ^ { 2 } \right] \right) ^ { \frac { 1 } { 2 } } d s . } \\ & { ^ { 3 } \big ( \mathbb { E } \left[ | X + Y | ^ { 2 } \right] \big ) ^ { \frac { 1 } { 2 } } \leq \big ( \mathbb { E } \left[ | X | ^ { 2 } \right] \big ) ^ { \frac { 1 } { 2 } } + \big ( \mathbb { E } \left[ | Y | ^ { 2 } \right] \big ) ^ { \frac { 1 } { 2 } } . } \end{array}
$$

## APPENDIX E PROOF OF PROPOSITION 2

Proof. For any $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { N }$ , because H(L) is symmetric,

$$
\langle \mathbf { H } ( \mathbf { L } ) ( \mathbf { x } - \mathbf { y } ) , \mathbf { x } - \mathbf { y } \rangle \leq \lambda _ { \operatorname* { m a x } } \bigl ( \mathbf { H } ( \mathbf { L } ) \bigr ) \| \mathbf { x } - \mathbf { y } \| ^ { 2 } .
$$

Since the eigenvalues of $\mathbf { H } ( \mathbf { L } )$ are $\{ h ( \lambda _ { i } ) \} _ { i = 1 } ^ { N }$ , then

$$
\lambda _ { \operatorname* { m a x } } \bigl ( \mathbf { H } ( \mathbf { L } ) \bigr ) = \operatorname* { m a x } _ { i \in \{ 1 , . . . , N \} } h ( \lambda _ { i } )
$$

and the result follows from the definition of one-sided Lipschitz constant. □

## APPENDIX F PROOF OF PROPOSITION 3

Proof. We first establish the lower bound. Evaluating the smoothness constraint at $\lambda = 2$ gives

$$
h ( 2 ) \leq \frac { 1 } { 2 } \ln \Big ( \frac { \eta } { 2 } \Big ) = - a .
$$

Since $h ( 0 ) = 0$ , evaluating the integral Lipschitz quotient (2) at $( \lambda _ { 1 } , \lambda _ { 2 } ) = ( 0 , 2 )$ gives

$$
C _ { \mathrm { I L } } ( h ) \geq \frac { 0 + 2 } { 2 ( 2 - 0 ) } | h ( 2 ) - h ( 0 ) | = \frac { | h ( 2 ) | } { 2 } \geq \frac { a } { 2 } .
$$

Thus, $C _ { K } ^ { \star } \geq { \frac { a } { 2 } } .$

For $K \geq 2$ and $b > 0 ,$ , define

$$
h _ { K , b } ( \lambda ) : = - b \left[ 1 - \left( 1 - \frac { \lambda } { 2 } \right) ^ { K } \right] , \qquad \lambda \in [ 0 , 2 ] .
$$

Function $h _ { K , b }$ is a polynomial of degree K, and it satisfies

$$
h _ { K , b } ( 0 ) = 0 , \qquad h _ { K , b } ( \lambda ) \leq 0 \quad \mathrm { f o r ~ a l l ~ } \lambda \in [ 0 , 2 ] .
$$

Claim. The integral Lipschitz constant of $h _ { K , b }$ is

$$
C _ { \mathrm { I L } } ( h _ { K , b } ) = \frac { b } { 2 } .\tag{27}
$$

Proof of the claim. Let

$$
f _ { K } ( x ) : = 1 - ( 1 - x ) ^ { K } .
$$

Since

$$
h _ { K , b } ( \lambda ) = - b f _ { K } \left( \frac { \lambda } { 2 } \right) ,
$$

setting

$$
x = \frac { \lambda _ { 1 } } { 2 } , \qquad y = \frac { \lambda _ { 2 } } { 2 } ,
$$

gives

$$
C _ { \mathrm { I L } } ( h _ { K , b } ) = b \operatorname* { s u p } _ { 0 \leq x < y \leq 1 } \frac { x + y } { 2 } \frac { f _ { K } ( y ) - f _ { K } ( x ) } { y - x } .
$$

For $0 \leq x < y \leq 1$ , define

$$
Q _ { K } ( x , y ) : = \frac { x + y } { 2 } \frac { f _ { K } ( y ) - f _ { K } ( x ) } { y - x } .
$$

We first want to show that for every $K \geq 2 ,$

$$
Q _ { K } ( x , y ) \leq { \frac { 1 } { 2 } } .\tag{28}
$$

To see this, write

$$
u : = 1 - \frac { x + y } { 2 } , \qquad v : = \frac { y - x } { 2 } .
$$

Then

$$
0 < u < 1 , \qquad 0 < v \leq \operatorname* { m i n } \{ u , 1 - u \} ,
$$

and

$$
Q _ { K } ( x , y ) = ( 1 - u ) \frac { ( u + v ) ^ { K } - ( u - v ) ^ { K } } { 2 v } .
$$

For fixed u,

$$
{ \frac { ( u + v ) ^ { K } - ( u - v ) ^ { K } } { 2 v } } = \sum _ { j = 1 \atop j { \mathrm { ~ o d d } } } ^ { K } { \binom { K } { j } } u ^ { K - j } v ^ { j - 1 } ,
$$

which is nondecreasing in v, since it contains only even powers of v with nonnegative coefficients.

If $0 < u \le 1 / 2$ , then $v \leq u$ , and therefore

$$
\begin{array} { c } { \displaystyle { Q _ { K } ( x , y ) \leq ( 1 - u ) \frac { ( 2 u ) ^ { K } } { 2 u } } } \\ { \displaystyle { = ( 1 - u ) 2 ^ { K - 1 } u ^ { K - 1 } \leq \frac { 1 } { 2 } , } } \end{array}
$$

where the last inequality follows because $( 1 - u ) u ^ { K - 1 }$ is nondecreasing on $( 0 , 1 / 2 ]$ for $K \geq 2$

If instead $1 / 2 < u < 1$ , then $v \leq 1 - u$ , and therefore

$$
\begin{array} { c } { \displaystyle Q _ { K } ( x , y ) \leq ( 1 - u ) \frac { 1 - ( 2 u - 1 ) ^ { K } } { 2 ( 1 - u ) } } \\ { \displaystyle = \frac { 1 - ( 2 u - 1 ) ^ { K } } { 2 } \leq \frac { 1 } { 2 } . } \end{array}
$$

Thus (28) holds for all $0 \leq x < y \leq 1$ . Moreover, $Q _ { K } ( 0 , 1 ) =$ $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ and therefore

$$
\operatorname* { s u p } _ { 0 \leq x < y \leq 1 } Q _ { K } ( x , y ) = \frac { 1 } { 2 } ,
$$

which proves (27).

□

It remains to choose b such that $h _ { K , b }$ is feasible, i.e.,

$$
h _ { K , b } ( \lambda ) \leq \frac { 1 } { 2 } \ln \left( \frac { \eta } { \lambda } \right) , \qquad \lambda \in ( 0 , 2 ] .
$$

For $\lambda \in ( 0 , \eta )$ , the right-hand side is positive, while $h _ { K , b } ( \lambda ) \leq$ 0, so the constraint is automatically satisfied. For $\lambda \in [ \eta , 2 ]$ the smoothness constraint can be written as

$$
- h _ { K , b } ( \lambda ) \geq \frac { 1 } { 2 } \ln \left( \frac { \lambda } { \eta } \right) = a + \frac { 1 } { 2 } \ln \left( \frac { \lambda } { 2 } \right) .
$$

Using the definition of $h _ { K , b } ,$ this gives

$$
b \geq \frac { a + \frac { 1 } { 2 } \ln \left( \frac { \lambda } { 2 } \right) } { 1 - \left( 1 - \frac { \lambda } { 2 } \right) ^ { K } } .
$$

The smallest b that guarantees feasibility for all $\lambda \in [ \eta , 2 ]$ is

$$
b _ { K } = \operatorname* { m a x } _ { \lambda \in \left[ \eta , 2 \right] } \frac { a + \frac { 1 } { 2 } \ln \left( \frac { \lambda } { 2 } \right) } { 1 - \left( 1 - \frac { \lambda } { 2 } \right) ^ { K } } .
$$

Evaluating at $\lambda = 2$ gives $b _ { K } \geq a$ . Moreover, for $\lambda \in [ \eta , 2 ]$

$$
a + \frac { 1 } { 2 } \ln \left( \frac { \lambda } { 2 } \right) \leq a ,
$$

and

$$
1 - \left( 1 - \frac { \lambda } { 2 } \right) ^ { K } \geq 1 - \left( 1 - \frac { \eta } { 2 } \right) ^ { K } = 1 - r ^ { K } .
$$

Hence $\begin{array} { r } { b _ { K } \le \frac { a } { 1 - r ^ { K } } } \end{array}$ . Thus, $h _ { K , b _ { K } }$ is feasible, and by (27),

$$
C _ { K } ^ { \star } \leq C _ { \mathrm { I L } } ( h _ { K , b _ { K } } ) = \frac { b _ { K } } { 2 } \leq \frac { a } { 2 ( 1 - r ^ { K } ) } .
$$

Combining this with the lower bound $C _ { K } ^ { \star } \geq a / 2$ gives

$$
\frac { a } { 2 } \leq C _ { K } ^ { \star } \leq \frac { a } { 2 ( 1 - r ^ { K } ) } ,
$$

which completes the proof.

## APPENDIX GPROOF OF PROPOSITION 4

Proof. Evaluating (24) at $x = 1$ gives

$$
b _ { K } \geq a .
$$

We derive a condition on K ensuring that $b _ { K } \leq a$ . From (24), it is sufficient that

$$
\frac { a + \frac 1 2 \ln x } { 1 - ( 1 - x ) ^ { K } } \leq a , \qquad x \in [ \eta / 2 , 1 ] .
$$

Equivalently, $2 a ( 1 - x ) ^ { K } \leq - \ln x$ . Using − ln x $\geq 1 - x ,$ it is sufficient that

$$
2 a ( 1 - x ) ^ { K - 1 } \leq 1 , \qquad x \in [ \eta / 2 , 1 ] .
$$

Since $1 - x \leq 1 - \eta / 2 = r ,$ , this is guaranteed by $2 a r ^ { K - 1 } \leq 1$ Taking logarithms and using ln $r < 0$ gives

$$
K \geq 1 + \frac { \ln ( 2 a ) } { - \ln r } .
$$

Thus, for integer $K \geq 2 ,$ it is sufficient that

$$
K \geq \operatorname* { m a x } \left\{ 2 , 1 + \left\lceil { \frac { \ln ( 2 a ) } { - \ln r } } \right\rceil \right\} = K _ { 0 } .
$$

Hence $b _ { K } \leq a$ . Together with $b _ { K } \geq a _ { \mathrm { ~ } }$ , this yields $b _ { K } = a ,$ which completes the proof. □

## APPENDIX H TOPOLOGICAL FLOW MATCHING

We recall some results from [17] that characterize the conditional paths and vector fields of linear Gaussian reference processes, and specialize them to the polynomial graph filters considered in this work.

Consider the reference process SDE

$$
d \mathbf { y } _ { t } = [ \mathbf { A } \mathbf { y } _ { t } + \pmb { \alpha } _ { t } ] d t + \sigma d \mathbf { w } _ { t } ,
$$

where A is a constant symmetric matrix and $\sigma > 0$ . As established in [17, Prop. 3], in the limit $\sigma  0$ , the bridge law conditioned on $\left( \mathbf { y } _ { 0 } , \mathbf { y } _ { 1 } \right)$ converges to a Dirac measure concentrated on its conditional mean,

$$
\mathbf { m } _ { t } ^ { \mathbf { y } _ { 0 } , \mathbf { y } _ { 1 } } = \mathbf { m } _ { t | 0 } ( \mathbf { y } _ { 0 } ) + \mathbf { R } _ { t , 1 | 0 } \mathbf { R } _ { 1 , 1 | 0 } ^ { - 1 } ( \mathbf { y } _ { 1 } - \mathbf { m } _ { 1 | 0 } ( \mathbf { y } _ { 0 } ) ) .\tag{29}
$$

Here, with $\mathbf { T } _ { s , t } = e ^ { ( t - s ) \mathbf { A } }$ denoting the state-transition matrix, the conditional mean and normalized covariance of the reference process, conditioned on $\mathbf { y } _ { 0 } .$ , are given by

$$
\mathbf { m } _ { t \mid 0 } ( \mathbf { y } _ { 0 } ) = \mathbf { T } _ { 0 , t } \mathbf { y } _ { 0 } + \int _ { 0 } ^ { t } \mathbf { T } _ { r , t } \mathbf { \alpha } \alpha _ { r } d r ,
$$

and

$$
{ \bf R } _ { t , s | 0 } = \int _ { 0 } ^ { \operatorname* { m i n } ( t , s ) } { \bf T } _ { r , t } { \bf T } _ { r , s } d r ,
$$

respectively. Furthermore, the conditional vector field in the zero-noise limit is given by [17, Prop. 2]

$$
u _ { t } ^ { \mathbf { y } _ { 0 } , \mathbf { y } _ { 1 } } = \mathbf { T } _ { t , 1 } \mathbf { R } _ { 1 , 1 | 0 } ^ { - 1 } ( \mathbf { y } _ { 1 } - \mathbf { m } _ { 1 | 0 } ( \mathbf { y } _ { 0 } ) ) .\tag{30}
$$

Regressing against the conditional vector field $u _ { t } ^ { \mathbf { y } _ { 0 } , \mathbf { y } _ { 1 } }$ under any coupling γ yields a valid flow-matching training objective, while the entropic optimal transport coupling [17, Prop. 4] recovers the solution to the degenerate Schrodinger bridge¨ problem [17, Prop. 1].

Polynomial graph filters. We specialize these results to $\mathbf { A } =$ $\mathbf { H } ( \mathbf { L } )$ , where

$$
\mathbf { H } ( \mathbf { L } ) = \sum _ { k = 0 } ^ { K } h _ { k } \mathbf { L } ^ { k } , \qquad h ( \boldsymbol { \lambda } ) = \sum _ { k = 0 } ^ { K } h _ { k } \boldsymbol { \lambda } ^ { k } ,
$$

and set $\alpha _ { t } = \mathbf { 0 }$ . Given the eigendecomposition $\mathbf { L } = \mathbf { V } \pmb { \Lambda } \mathbf { V } ^ { \top }$ the graph filter H(L) shares the eigenvectors V of L, with eigenvalues $\{ h ( \lambda _ { i } ) \} _ { i = 1 } ^ { N }$ . Defining $h _ { i } = h ( \lambda _ { i } )$ and the GFT $\hat { \mathbf { y } } = \mathbf { V } ^ { \top } \mathbf { y }$ , the state-transition matrix becomes

$$
\mathbf { T } _ { s , t } = \mathbf { V } \mathrm { d i a g } ( e ^ { h _ { i } ( t - s ) } ) \mathbf { V } ^ { \top } .
$$

Consequently, the normalized covariance is diagonal in the graph Fourier basis,

$$
\mathbf { V } ^ { \top } \mathbf { R } _ { t , s | 0 } \mathbf { V } = \mathrm { d i a g } ( q _ { i } ( t , s ) ) ,
$$

where

$$
q _ { i } ( t , s ) = \int _ { 0 } ^ { \operatorname* { m i n } ( t , s ) } e ^ { h _ { i } ( t + s - 2 r ) } d r .
$$

In particular,

$$
\mathbf { V } ^ { \top } \mathbf { R } _ { t , 1 | 0 } \mathbf { R } _ { 1 , 1 | 0 } ^ { - 1 } \mathbf { V } = \mathrm { d i a g } \left( \frac { q _ { i } ( t , 1 ) } { q _ { i } ( 1 , 1 ) } \right) = \mathrm { d i a g } ( c _ { t , i } ) ,
$$

with

$$
c _ { t , i } = \left\{ \begin{array} { l l } { e ^ { h _ { i } ( 1 - t ) } \displaystyle \frac { e ^ { 2 h _ { i } t } - 1 } { e ^ { 2 h _ { i } } - 1 } , } & { h _ { i } \neq 0 , } \\ { t , } & { h _ { i } = 0 . } \end{array} \right.
$$

Substituting into (29) yields the deterministic conditional path

$$
\hat { y } _ { t , i } = e ^ { h _ { i } t } \hat { y } _ { 0 , i } + c _ { t , i } ( \hat { y } _ { 1 , i } - e ^ { h _ { i } } \hat { y } _ { 0 , i } ) .\tag{31}
$$

Similarly, specializing (30) gives the conditional vector field

$$
\hat { u } _ { t , i } ^ { \mathbf { y } _ { 0 } , \mathbf { y } _ { 1 } } = a _ { t , i } ( \hat { y } _ { 1 , i } - e ^ { h _ { i } } \hat { y } _ { 0 , i } ) ,\tag{32}
$$

where

$$
a _ { t , i } = \left\{ \begin{array} { l l } { { e ^ { h _ { i } ( 1 - t ) } \displaystyle \frac { 2 h _ { i } } { e ^ { 2 h _ { i } } - 1 } , } } & { { h _ { i } \neq 0 , } } \\ { { 1 , } } & { { h _ { i } = 0 . } } \end{array} \right.
$$

These expressions generalize the heat-equation conditional paths of [17], recovered by setting $h _ { i } = - \kappa \lambda _ { i } ,$ , to arbitrary polynomial graph filters. The conditional paths and vector fields derived here are transformed back to the vertex domain using the inverse GFT and employed to train the learned vector field $u _ { t } ^ { \theta }$ , as tabulated in Algorithm 1.

[1] M. Schmidt and G. Mateos, “Stable filters for generative modeling of graph signals,” https://arxiv.org/abs/2609.18759, 2026.

[2] M. Welling, S. Lu, and L. Holdijk, Generative AI and Stochastic Thermodynamics: A Tale of Free Energies. Cambridge University Press, 2026.

[3] Y. Xie and X. Cheng, “Flow-based generative models as iterative algorithms in probability space: An intuitive mathematical framework,” IEEE Signal Process. Mag., vol. 43, no. 2, pp. 37–50, 2026.

[4] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in Proc. Int. Conf. Learn. Representations (ICLR), 2023, pp. 1–28.

[5] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Proc. Adv. Neural. Inf. Process. Syst. (NeurIPS), 2020, pp. 1–12.

[6] R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud, “Neural ordinary differential equations,” in Proc. Adv. Neural. Inf. Process. Syst. (NeurIPS), 2018, pp. 1–13.

[7] J. Liu, A. Kumar, J. Ba, J. Kiros, and K. Swersky, “Graph normalizing flows,” in Proc. Adv. Neural. Inf. Process. Syst. (NeurIPS), 2019, pp. 1–11.

[8] C. Niu, Y. Song, J. Song, S. Zhao, A. Grover, and S. Ermon, “Permutation invariant graph generation via score-based generative modeling,” in Proc. Int. Conf. Artif. Intell. Statist. (AISTATS), 2020, pp. 1–10.

[9] J. Jo, S. Lee, and S. J. Hwang, “Score-based generative modeling of graphs via the system of stochastic differential equations,” in Proc. Int. Conf. Mach. Learn. (ICML), 2022, pp. 10 362–10 383.

[10] C. Liu, W. Fan, Y. Liu, J. Li, H. Li, H. Liu, J. Tang, and Q. Li, “Generative diffusion models on graphs: Methods and applications,” in Proc. Int. Joint Cont. Artificial Intell., 2023, pp. 6702–6711.

[11] C. Vignac, I. Krawczuk, A. Siraudin, B. Wang, V. Cevher, and P. Frossard, “Digress: Discrete denoising diffusion for graph generation,” in Proc. Int. Conf. Learn. Representations (ICLR), 2023, pp. 1–22.

[12] H. Chen, N. Zilberstein, and S. Segarra, “Prior-informed flow matching for graph reconstruction,” in ICML Wrksp. Struct. Prob. Inference and Gen. Modeling, 2026, pp. 1–33.

[13] Y. Zhu, C. Wang, Q. Zhang, and H. Xiong, “Graph signal diffusion model for collaborative filtering,” in Proc. ACM SIGIR Conf. Res. Develop. Inf. Retrieval, 2024, pp. 1380–1390.

[14] Y. B. Uslu, S. Hadou, S. Rozada, S. S. Bidokhti, and A. Ribeiro, “Generative diffusion models of stochastic graph signals,” arXiv preprint arXiv:2607.06833, 2026.

[15] S. Rozada, Vimal K B, A. Cavallo, A. G. Marques, H. Jamali-Rad, and E. Isufi, “Graph-aware diffusion for signal generation,” in Proc. IEEE Intl. Conf. Acoustics, Speech and Signal Process. (ICASSP), 2026, pp. 461–465.

[16] M. Yang, “Topological Schrodinger bridge matching,” in ¨ Proc. Int. Conf. Learn. Representations (ICLR), vol. 2025, 2025, pp. 1–42.

[17] K. Wyrwal, I. I. Ceylan, and A. Tong, “Topological flow matching,” in Proc. Int. Conf. Learn. Representations (ICLR), vol. 2026, 2026, pp. 1–26.

[18] M. Schmidt, S. Silva, F. Larroca, G. Mateos, and P. Muse, “Graph´ contrastive learning for connectome classification,” in Proc. Asilomar Conf. Signals, Syst., Computers, 2025, pp. 1027–1032.

[19] B. Yu, H. Yin, and Z. Zhu, “Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting,” in Proc. Int. Joint Cont. Artificial Intell. (IJCAI), 2018, pp. 3634–3640.

[20] R. G. Camargo, Z. Wang, and A. Ribeiro, “Scalability of graph neural network policies in wireless communication networks,” arXiv preprint arxiv:2609.06066, 2026.

[21] V. De Bortoli, J. Thornton, J. Heng, and A. Doucet, “Diffusion Schrodinger bridge with applications to score-based generative mod-¨ eling,” in Proc. Adv. Neural. Inf. Process. Syst. (NeurIPS), 2021, pp. 1–15.

[22] T. Chen, G.-H. Liu, and E. Theodorou, “Likelihood training of Schrodinger bridge using forward-backward SDEs theory,” in¨ Proc. Int. Conf. Learn. Representations (ICLR), 2022, pp. 1–27.

[23] G.-H. Liu, Y. Lipman, M. Nickel, B. Karrer, E. Theodorou, and R. T. Chen, “Generalized Schrodinger bridge matching,” in¨ Proc. Int. Conf. Learn. Representations (ICLR), 2024, pp. 1–26.

[24] E. D. Kolaczyk, Statistical Analysis of Network Data: Methods and Models. New York, NY: Springer, 2009.

[25] A. Zalesky, A. Fornito, and E. Bullmore, “On the use of correlation as a measure of network connectivity,” Neuroimage, vol. 60, no. 4, pp. 2096–2106, 2012.

[26] G. Mateos, S. Segarra, A. G. Marques, and A. Ribeiro, “Connecting the dots: Identifying network structure via graph signal processing,” IEEE Signal Process. Mag., vol. 36, no. 3, pp. 16–43, May 2019.

[27] X. Dong, D. Thanou, M. Rabbat, and P. Frossard, “Learning graphs from data: A signal representation perspective,” IEEE Signal Process. Mag., vol. 36, no. 3, pp. 44–63, 2019.

[28] M. M. Bronstein, J. Bruna, T. Cohen, and P. Velickoviˇ c, “Geometric´ Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges,” arXiv preprint arXiv:2104.13478, 2021.

[29] F. Gama, J. Bruna, and A. Ribeiro, “Stability properties of graph neural networks,” IEEE Trans. Signal Process., vol. 68, p. 5680–5695, 2020.

[30] L. Ruiz, L. F. O. Chamon, and A. Ribeiro, “Transferability properties of graph neural networks,” IEEE Trans. Signal Process., vol. 71, pp. 3474 – 3489, 2023.

[31] H. Kenlay, D. Thanou, and X. Dong, “Interpretable stability bounds for spectral graph filters,” in Proc. Int. Conf. Mach. Learn. (ICML), 2021, pp. 5388–5397.

[32] R. Levie, W. Huang, L. Bucci, M. Bronstein, and G. Kutyniok, “Transferability of spectral graph convolutional neural networks,” J. Mach. Learn. Res., vol. 22, pp. 1–59, 2021.

[33] N. Keriven, A. Bietti, and S. Vaiter, “Convergence and stability of graph convolutional networks on large random graphs,” in Proc. Adv. Neural. Inf. Process. Syst. (NeurIPS), vol. 33, 2020, pp. 21 512–21 523.

[34] M. Schmidt and G. Mateos, “Stability of flow models for graph signals,” arXiv preprint arXiv:2607.07510, 2026.

[35] M. Yan, C. Kulick, and S. Tang, “On the convergence and size transferability of continuous-depth graph neural networks,” https://arxiv.org/abs/2510.03923, 2026.

[36] M. Poli, S. Massaroli, J. Park, A. Yamashita, H. Asama, and J. Park, “Graph neural ordinary differential equations,” https://arxiv.org/abs/1911.07532, 2021.

[37] A. Ortega, P. Frossard, J. Kovacevi ˇ c, J. M. Moura, and P. Vandergheynst,´ “Graph signal processing: Overview, challenges, and applications,” Proc. IEEE, vol. 106, no. 5, pp. 808–828, 2018.

[38] A. Sandryhaila and J. M. F. Moura, “Discrete signal processing on graphs: Frequency analysis,” IEEE Trans. Signal Process., vol. 62, no. 12, pp. 3042–3054, 2014.

[39] E. Isufi, F. Gama, D. I. Shuman, and S. Segarra, “Graph filters for signal processing and machine learning on graphs,” IEEE Trans. Signal Process., vol. 72, pp. 4745–4781, 2024.

[40] F. Gama, A. G. Marques, G. Leus, and A. Ribeiro, “Convolutional neural network architectures for signals supported on graphs,” IEEE Trans. Signal Process., vol. 67, no. 4, p. 1034–1049, 2019.

[41] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. Int. Conf. Learn. Representations (ICLR), 2017, pp. 1–14.

[42] M. Defferrard, X. Bresson, and P. Vandergheynst, “Convolutional neural networks on graphs with fast localized spectral filtering,” in Proc. Adv. Neural. Inf. Process. Syst. (NeurIPS), 2016, pp. 1–9.

[43] L. Ruiz, F. Gama, and A. Ribeiro, “Graph neural networks: Architectures, stability and transferability,” Proc. IEEE, vol. 109, no. 5, pp. 660– 682, 2021.

[44] C. Leonard, “A survey of the Schr´ odinger problem and some of its¨ connections with optimal transport,” Discrete & Continuous Dynamical Systems - A, vol. 34, no. 4, pp. 1533–1574, 2014.

[45] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in Proc. Int. Conf. Learn. Representations (ICLR), 2021, pp. 1–36.

[46] A. Q. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic models,” in Proc. Int. Conf. Mach. Learn. (ICML), 2021, pp. 8162–8171.

[47] J.-C. Delvenne, R. Lambiotte, and L. E. Rocha, “Diffusion on networked systems is a question of time or structure,” Nature communications, vol. 6, no. 1, p. 7366, 2015.

[48] A. V. Nikitin, S. John, A. Solin, and S. Kaski, “Non-separable spatiotemporal graph kernels via spdes,” in Proc. Int. Conf. Artif. Intell. Statist. (AISTATS), 2022, pp. 10 640–10 660.

[49] R. Hettich and K. O. Kortanek, “Semi-infinite programming: Theory, methods, and applications,” SIAM J. Optim., vol. 35, no. 3, pp. 380– 429, 1993.

[50] P. W. Holland, K. B. Laskey, and S. Leinhardt, “Stochastic blockmodels: First steps,” Social Networks, vol. 5, no. 2, pp. 109–137, 1983.

[51] Human Connectome Project, https://www.humanconnectome.org/.