![](images/51fa8a1e5663c0ece3bf24f965a71080c3b630fa4294a63e4418101e1c87ea94.jpg)  
Figure 1: Visualization of ghost tasking as the minimal amount of tasks necessary to depict any linear system of differential equations through parametrization. On top, we show the equations each system is solving, in the middle we show the forward fit of a physical system and below, the the mean absolute error of the two learned physical parameters and their ratio relative to the true values in an inverse problem.

## Graphical Abstract

Ghost tasking for parametrized Gaussian Processes solving linear differential equations Johanna Moser, Christopher Albert, Sascha Ranftl

## Highlights

Ghost tasking for parametrized Gaussian Processes solving linear differential equations Johanna Moser, Christopher Albert, Sascha Ranftl

• Ghost tasking balances accuracy and applicability in physics-informed GPs

• We connect the physics-informed GP methods of Raissi (2017) to Lange-Hegermann (2018)

• We prove ghost tasking valid for full-row-rank systems with meromorphic coefficients

• Ghost tasking improves accuracy over Raissi (2017) in low-data inverse problems

• Computer algebra tutorials included for OreModules and Macaulay2

# Ghost tasking for parametrized Gaussian Processes solving linear differential equations

Johanna Mosera,\*, Christopher Alberta, Sascha Ranftlb,c,1

ªInstitute of Theoretical & Computational Physics, University of Technology, Graz, Austria bCourant Institute, New York University, New York, USA cDivision of Applied Mathematics, Brown University, USA

## Abstract

Physics-informed machine learning has gained significant attention in recent years. In regimes of limited data, parametrized Gaussian processes have become popular. Existing approaches, however, often face limitations, such as requiring parametrizable (also called controllable) systems or a large number of output tasks. In this work, we introduce a systematic procedure we call "ghost tasking", using auxiliary tasks to circumvent these limitations. We prove that such ghost tasks can render any non-parametrizable system effectively parametrizable, enabling algorithmic construction of parametrized Gaussian Processes while keeping the number of required tasks (i.e. output dimensions) and latent functions low. We find that ghost tasking performs especially well in an inverse problem setting, even with very few available data. We show the usage and power of ghost tasking in three experiments, providing systematic comparisons to the only other currently available method applicable to all experiments. We provide necessary syntax and explications for two computer algebra programs that compute parametrizations for systems with polynomial or rational coefficients. Our theoretical results extend to systems with meromorphic functions.

Keywords: Physics-informed machine learning, Gaussian Processes, Homological algebra, Parametrizability, Probabilistic machine learning, Ghost tasking

## 1. Introduction

Understanding complex systems from data typically relies on either data-driven models or physics-based formulations. Hybrid approaches have emerged under the name physics-informed machine learning, aiming to integrate physical structure directly into learning systems. A central design choice in such hybrid models is how physical laws are enforced. Soft constraint approaches incorporate differential equations as penalties, whereas hard constraint methods enforce them exactly. In this work, we focus on hard constraints within the Gaussian process (GP) framework, which guarantees exact solutions of linear differential equations through the structure of the covariance function (also called kernel) while retaining principled uncertainty quantification capabilities.

## 1.1. Related Work

Research on incorporating physically motivated constraints into Gaussian processes has accelerated considerably in recent years; a survey of methods as of 2020 can be found in [1]. We provide here a brief and non-exhaustive orientation.

The first class of methods capable of enforcing hard constraints are parametrized Gaussian Processes, which we will focus on in this work. They construct covariance functions by exploiting the closure of Gaussian processes under linear transformations, applying linear differential operators directly to base kernel functions. The earliest works in this direction are [2, 3, 4] which have subsequently been applied in [5, 6, 7] and rendered algorithmic in [8, 9, 10, 11, 12].

Apart from the parametrization-based methods, another line of work that enforces linear differential equations exactly is represented by Mercer-type GPs whose kernels are directly built from fundamental solutions of the differential equations at hand. This approach has been proposed in [13] and has been re-discovered and applied to different equations repeatedly since [14, 15]. They however have the disadvantage that we need to know the fundamental solution of the specific problem at hand beforehand, making them, in contrast to parametrization-based GPs, non-algorithmic.

Similar approaches have been proposed to enforce boundary conditions through specific kernel choices and/or mean-function constructions [16, 17]. Harmonic-feature representations based on Laplacian eigenfunctions on irregular domains [18] have also been proposed with the additional advantage of providing computationally efficient low-rank GP representations. Fulfilling boundary conditions can also be combined with fulfilling differential equations inside of the domains and has been explored for example in [19, 20].

Finally, especially for nonlinear PDEs, physics is often incorporated during conditioning as a soft constraint rather than through exact prior construction. Numerical Gaussian process methods using Euler integration schemes and collocation-based formulations enforce differential equations at selected points and have been applied to a broad range of nonlinear problems [21, 22, 23], while recent work has further generalized GP conditioning to flexible classes of constraints and observations [24].

## 1.2. Contribution

The present work focuses on parametrization-based GP models that incorporate linear differential operators in the covariance function. Parametrization-based methods provide the strongest combination of exact constraint satisfaction and algorithmic construction, yet they remain subject to important limitations. The method of Raissi et al. [9] is applicable to a broad class of linear differential equations, but can lack robustness in low-data regimes [25]. More efficient techniques however rely on restrictive assumptions such as parametrizability [10], or the coefficients in the differential equations being constant [11, 12]. We address these limitations by introducing ghost tasking, a construction that generalizes parametrization-based GP models to traditionally not parametrizable systems, including those with non-constant coefficients, while preserving performance in forward and inverse problems. Therein, ghost tasks are an auxiliary (ghost) dimension in the GP output; i.e. the a non-parametrizable system is lifted to a higher dimension where it becomes parametrizable. After parametrizing the system in this higher dimension, we can conveniently convert the GP back to the original lower dimension by conditioning the ghost tasks to zero.

A detailed conceptual comparison with existing methods is deferred to chapter 5.

The main idea of ghost tasking is to (1) embed an non-parametrizable system represented by its operator matrix in a higher-dimensional space by adding ghost tasks, (2) compute the now fully parametrizable null space of the embedded operator matrix, (3) apply the null space as a linear operator to parametrize a base kernel of a GP and (4) re-constrain the embedded parametrized GP to the original solution space by conditioning the ghost tasks to be point wise zero. As discussed in [10], the computation of operator null spaces can be performed very efficiently with computer algebra programs. Two open source packages that work well are OreModules [26] and Macaulay2 [27].

Our contributions are the following:

1. We identify the work of Raissi et al. (2017) [9] and Lange-Hegermann (2018)[10] regarding “physics-informed GPs"and GPs for differential equations more generally as belonging to the same methodological spectrum.

2. We introduce ghost tasking as the optimum on said spectrum, bridging the gap between accuracy and applicability.

3. We prove the theoretical validity of ghost tasking for all systems with full row-rank, and polynomial, rational, or generally meromorphic coefficients

4. We show improved accuracy, especially in low data settings, for inverse problems in 3 experiments and systematic comparisons to Raissi et al. [9]

5. Our method is fully compatible with the already freely available python package PCGP [25] and we include tutorials for using ghost tasking with the computer algebra programs OreModules [26] and Macaulay2 [27].

The remaining paper will be structured in the following way: in chapter 2 we first introduce Gaussian Processes and subsequently discuss two state of the art methods for differential equations with non-constant coefficients to parametrize GPs [9, 10] in chapter 2.2. In chapter 2.3, we give an introduction to the algebraic concepts that we use in chapter 2.4 to analyze the methods introduced in chapter 2.2 with regards to parametrizability. Our theoretical contributions are introduced in chapter 3.2. In chapter 3.1, we show how [9] and [10] are related and use these findings to develop ghost tasking in chapter 3.2. We analyze ghost tasking theoretically in chapter 3.3. In chapter 4 we show three different experiments applying ghost tasking. Each of the experiments contains an algebraic part where we show the application of ghost tasking, followed by a numerical implementation and evaluation of the problem in a forward and inverse setting, and systematic performance comparison to [9], since this is the only available GP parametrization method next to ghost tasking that is general enough so that it can be applied to all proposed experiments. Finally, we discuss practical aspects of ghost tasking and place it within the context of the landscape of currently available methods in chapter 5. In Appendix A we provide and explain code snippets for the practical use of the computer algebra programs OreModules [26] and Macaulay2 [27] for ghost tasking. Ablation studies on the ghost tasking method can be found in Appendix B, and an example for commutative cases as discussed in chapter 5 is provided in Appendix C.

## 2. Background

## 2.1. Gaussian Processes

Gaussian Processes are a flexible framework for regression function approximation that naturally incorporates uncertainty. For a comprehensive introduction, see e.g. [28]. Given an input set $\boldsymbol { \chi } _ { \mathbf { \Lambda } \subset \mathbf {  { \mathbb { R } } ^ { d } } }$ with elements $\mathbf { x } , \mathbf { x ^ { \prime } } \in X ,$ a GP $g \sim \mathcal { G P } ( m ( \mathbf { x } ) , k ^ { \vartheta } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) )$ is defined by a mean function m(x) and a kernel function $k ^ { \vartheta } ( \mathbf { x } , \mathbf { x } ^ { \prime } )$ with hyperparameters $\vartheta .$ For any finite set of inputs $\chi ^ { n } \ni X = ( \mathbf { x } _ { 1 } , . . . , \mathbf { x } _ { n } )$ , g(X) follows a multivariate normal distribution with mean ${ \vec { \mu } } = m ( X )$ whose entries are $\mu _ { i } = m ( \mathbf { x } _ { i } )$ and covariance matrix Σ frequently written as $k ^ { \vartheta } ( X , X )$ since the entries of the covariance matrix are defined through the kernel function as $\Sigma _ { i j } = k ^ { \vartheta } ( { \bf x } _ { i } , { \bf x } _ { j } )$ . Given n noisy observations $\boldsymbol { Y } = ( y _ { 1 } , \cdots , y _ { n } ) ^ { T }$ with $y _ { i } = g ( \mathbf { x } _ { i } ) + \varepsilon _ { i }$ at the points $X$ with independently distributed measurement noise terms $\varepsilon _ { i } \sim { \cal N } ( 0 , \sigma ^ { 2 } )$ , conditioning yields again $\mathbf { a } \operatorname { G P }$ whose mean and covariance matrix at input new points $X ,$ are obtained by the standard multivariate Gaussian conditioning formulas

$$
\begin{array} { l } { { m _ { | X , Y } ( X _ { * } ) = m ( X _ { * } ) + k ^ { \vartheta } ( X _ { * } , X ) \left( k ^ { \vartheta } ( X , X ) + \sigma ^ { 2 } \mathbb { I } \right) ^ { - 1 } ( Y - m ( X ) ) } } \\ { { k _ { | X , Y } ( X _ { * } , X _ { * } ) = k ^ { \vartheta } ( X _ { * } , X _ { * } ) - k ^ { \vartheta } ( X _ { * } , X ) ( k ^ { \vartheta } ( X , X ) + \sigma ^ { 2 } \mathbb { I } ) ^ { - 1 } k ^ { \vartheta } ( X , X _ { * } ) . } } \end{array}\tag{1}
$$

The kernel encodes prior beliefs about our function such as smoothness, differentiability, and, through $\vartheta ,$ hyperparameters such as relevant lengthscales. The hyperparameters 9 are generally chosen by minimizing the negative log marginal likelihood, i.e. for a zero-mean prior

$$
- \log { \left( p ( Y | \vartheta , X , \sigma ) \right) } = \frac { 1 } { 2 } Y ^ { T } ( k ^ { \vartheta } ( X , X ) + \sigma ^ { 2 } \mathbb { I } ) ^ { - 1 } Y + \frac { 1 } { 2 } \log { \left( \operatorname* { d e t } ( k ^ { \vartheta } ( X , X ) + \sigma ^ { 2 } \mathbb { I } ) \right) } + \frac { n } { 2 } \log ( 2 \pi )\tag{2}
$$

consisting of a data-fit term $( Y ^ { T } ( k ^ { \vartheta } ( X , X ) + \sigma ^ { 2 } \mathbb { I } ) ^ { - 1 } Y )$ , a complexity penalty $\mathrm { ( l o g ( d e t } ( k ^ { \vartheta } ( X , X ) +$ $\sigma ^ { 2 } \mathbb { I } ) ) )$ , and a constant term to normalize the likelihood. If we have prior beliefs about the hyperparameters $p ( \vartheta )$ we can simply $\operatorname { a d d } - \log ( p ( \vartheta ) )$ to (2) to obtain the maximum a posteriori $( \mathrm { M A P } )$ estimates of 9. Just like their finite dimensional analogues, multivariate Gaussian distributions, Gaussian Processes are also closed under linear transformations T, i.e. linearly transforming the output of a GP corresponds to sampling from another GP whose mean and kernel are transformed, formally

$$
T g \sim \mathcal { G P } ( T m ( \mathbf { x } ) , T k ^ { \vartheta } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) T ^ { \prime } )\tag{3}
$$

where $T$ acts on x and $T ^ { \prime }$ acts on $\mathbf { x } ^ { \prime }$

For multidimensional outputs, we can define the so-called "multitask $\mathrm { G P " }$ by stacking the outputs (also called "tasks") in the mean vector and covariance matrix. This multitask structure and $\mathrm { G P s } ^ { \mathrm { \prime } }$ closedness under linear transforms gives us the flexibility to build GPs whose outputs are coupled by a given linear system of differential equations, rather than treating each output as an independent GP. This allows us to encode physical constraints directly into the kernel of the model.

## 2.2. Parametrized Gaussian Processes

We now discuss two particular state-of-the-art methods for incorporating linear differential equations into GPs. To our knowledge, they are the only two methods in the literature that are both algorithmic and capable of parametrizing GPs for learning solutions of differential equations with non-constant coefficients. We use the umbrella term "parametrized $\mathrm { G P s " }$ to refer to them collectively. A linear system of differential equations can be written as

$$
{ \mathcal { L } } ^ { \phi } ( \mathbf { x } ) { \vec { u } } ( \mathbf { x } ) = { \vec { f } } ( \mathbf { x } ) \quad { \mathrm { f o r ~ a l l ~ } } \mathbf { x } \in { \mathcal { X } } .\tag{4}
$$

Here, ${ \vec { u } } ( \mathbf { x } )$ is a vector containing all unknown functions $( \mathrm { i . e . }$ the solution), and ${ \vec { f } } ( \mathbf { x } )$ contains the inhomogeneities. $\mathcal { L } ^ { \phi } ( \mathbf { x } )$ is a matrix containing a set of physical parameters $\phi$ and differential operators $\partial _ { \mathbf { x } }$ with coefficient functions depending on x.

Differential operators acting on functions of variables $x _ { 1 } , . . . , x _ { n }$ can be organized into algebraic rings. Depending on the class of coefficient functions allowed, one obtains different rings – for example rings of differential operators with meromorphic or rational coefficient functions. Of particular practical interest is the case of polynomial coefficients, leading to Weyl algebras, which are especially convenient for computer algebra. We will use them for our experiments in section 4. For describing differential equations in n variables, we define the n-th Weyl algebra $A _ { n } ( \mathbb { R } )$ as consisting of elements that are finite linear combinations of terms $\Pi _ { i = 1 } ^ { n } x _ { i } ^ { k _ { i } } \partial _ { i } ^ { l _ { i } }$ with $k _ { i } , l _ { i } \in \mathbb { N }$ where the generators satisfy the commutation relations

$$
\begin{array} { c } { { [ \partial _ { i } , x _ { i } ] _ { - } : = \partial _ { i } x _ { i } - x _ { i } \partial _ { i } = 1 \mathrm { ~ f o r ~ a l l ~ } i = 1 , . . . . n } } \\ { { { } [ x _ { i } , x _ { j } ] _ { - } = [ \partial _ { i } , \partial _ { j } ] _ { - } = 0 \mathrm { ~ f o r ~ a l l ~ } i , j } } \\ { { { } [ \partial _ { j } , x _ { i } ] _ { - } = 0 \mathrm { ~ f o r ~ } i \not = j , } } \end{array}\tag{5}
$$

i.e. $\partial _ { i }$ acts as differentiation with respect to x. A similar structure can be used for rational coefficients, giving the ring $B _ { n } ( \mathbb { R } )$ ).

All results in this paper apply to a class of noncommutative rings that are are Noetherian domains, so-called "very simple domains", and have finite global dimension [29]. Concretely, we state the results for Weyl algebras $A _ { n } ( \mathbb { R } )$ and $B _ { n } ( \mathbb { R } )$ (for polynomial and rational coefficients), rings with meromorphic coefficients, and, in the ODE case, rings with holomorphic (formal or locally convergent power series) coefficients. We will simply denote such a ring by D throughout the paper unless stated otherwise.

The most prominent version to parametrize a Gaussian Process using operators in D was introduced by [9]. We will briefly summarize the method here: Let us assume the unknown functions  in Eq. 4 can be expressed through a "base $\mathrm { G P " }$ or "latent $\mathrm { G P " } u ( \mathbf { x } ) \sim \mathcal { G P } ( m _ { u } ( \mathbf { x } ) , k _ { u u } ( \mathbf { x } , \mathbf { x } ) )$ In general, this base $\mathrm { G P }$ has a base mean $m _ { u } ( { \bf x } )$ conventionally chosen to be zero and a sufficiently continuously differentiable base kernel $k _ { u u } ( \mathbf { x } , \mathbf { x } ^ { \prime } )$ . A common choice for the latter is the squaredexponential kernel $\begin{array} { r } { k _ { u u } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = A \exp \left( - \frac { \vert \vert \mathbf { x } - \mathbf { x } ^ { \prime } \vert \vert ^ { 2 } } { 2 l } \right) ~ } \end{array}$ that is infinitely often differentiable and has two hyperparameters θ, the lengthscale l and the scale factor denoted by A. If $\vec { u } ( \mathbf { x } )$ has more than one output dimension, the base kernel can consist of multiple independent square-exponential kernels stacked in multitask fashion, with either shared or independent hyperparameter values. We can then exploit the linearity of $\mathcal { L } ^ { \phi } ( \mathbf { x } )$ and model (x) and the inhomogeneities ${ \vec { f } } ( \mathbf { x } )$ together as a multitask $\mathrm { G P }$ of the form

$$
\begin{array} { r l } &  \binom { \vec { u } ( \mathbf { x } ) } { \vec { f } ( \mathbf { x } ) } \sim \mathcal { G P } \left( \vec { 0 } , \left( \begin{array} { c c } { k _ { u u } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) } & { \mathcal { L } ^ { \phi T } ( \mathbf { x } ^ { \prime } ) k _ { u u } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) } \\ { \mathcal { L } ^ { \phi } ( \mathbf { x } ) k _ { u u } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) } & { \mathcal { L } ^ { \phi } ( \mathbf { x } ) k _ { u u } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) \mathcal { L } ^ { \phi T } ( \mathbf { x } ^ { \prime } ) \right) \right) , } \end{array} \end{array}\tag{6}
$$

where the transpose $\mathcal { L } ^ { \phi T } ( \mathbf { x } ^ { \prime } )$ is applied to the second argument of the base kernel. By conditioning on data points of ${ \vec { f } } ( \mathbf { x } )$ we can infer $\vec { u } ( \mathbf { x } )$ , which is called a forward problem, and if we know data from both ${ \vec { f } } ( \mathbf { x } )$ and $\vec { u } ( \mathbf { x } )$ , we can learn the probability distribution of the physical parameters φ through hyperparameter optimization or sampling, which is referred to as the inverse problem. In analogy to physics-informed neural networks [30], we will refer to this method as "physicsinformed Gaussian Processes" (PIGPs) and use them as comparison baseline for our method.

A downside of PIGPs is that they require a lot of tasks: for a system with $q$ unknown functions we need to model both the functions and their inhomogeneities, so we need $2 q$ tasks, which is unfavorable in terms of the scaling of the computational effort with the number of dimensions. Accordingly, we also need q independent base kernels, which makes the latent function space for (x) oftentimes unnecessarily large and thus requires more data to be sufficiently constrained. [8, 10] introduced an approach that is able to encode a system in a more compact way, making the latent function space smaller and rendering the approach more data efficient and more accurate in the inverse problem ([25]). To explain this method, we will rewrite (4) in the form:

$$
R ^ { \phi } ( { \bf x } ) \vec { \xi } ( { \bf x } ) = 0 \quad \mathrm { f o r ~ a l l ~ } { \bf x } \in \boldsymbol { \cal X } ,\tag{7}
$$

where $R ^ { \phi } ( { \bf x } ) \ \in \ D ^ { q \times p }$ is called the system matrix. As before, $\phi$ denotes physical parameters and the elements of $R ^ { \phi } ( \mathbf { x } )$ lie in an operator ring D. Both the unknown functions and any inhomogeneities are components of the p-dimensional vector-valued output function $\vec { \xi } ( \mathbf { x } )$ with $p \geq q .$

The method of Lange-Hegermann [10] proposes to use the null space of $R ^ { \phi } ( \mathbf { x } )$ in the operator space D to solve the differential equation: If $Q ^ { \phi } ( \mathbf { x } ) \in D ^ { p \times n }$ denotes the null space of $R ^ { \phi } ( \mathbf { x } )$ , i.e. the matrix whose n column vectors $\dot { Q } _ { . . i } ^ { \phi } ( \mathbf { x } )$ fulfill $R ^ { \phi } ( { \bf x } ) { \cal Q } _ { . . i } ^ { \phi } ( { \bf x } ) = 0$ , then for any set of (sufficiently differentiable) functions $g _ { i }$ , the function $\begin{array} { r } { \vec { \xi } ( \mathbf { x } ) : = \sum _ { i = 1 } ^ { n } { Q _ { \cdot , i } ^ { \phi } ( \mathbf { x } ) g _ { i } } = Q ^ { \phi } ( \mathbf { x } ) \vec { g } ( \mathbf { x } ) } \end{array}$ automatically fulfills

$$
R ^ { \phi } ( { \bf x } ) \vec { \xi } ( { \bf x } ) = R ^ { \phi } ( { \bf x } ) ( Q ^ { \phi } ( { \bf x } ) \vec { g } ( { \bf x } ) ) = \big ( R ^ { \phi } ( { \bf x } ) Q ^ { \phi } ( { \bf x } ) \big ) \vec { g } ( { \bf x } ) = 0 \vec { g } ( { \bf x } ) = 0 ,
$$

and is therefore a solution to $( 7 )$

Based on this observation, the author of [10] shows that we can build a Gaussian Process whose realizations $\xi ( \mathbf { x } )$ are dense in the solution space of Eq. 7 by transforming a base GP model (x) with this null space operator $Q ^ { \phi } ( \mathbf { x } )$ . We will refer to this method as "null space-based" (NSB) method. The NSB Gaussian Process has the form

$$
\vec { \xi } ( \mathbf { x } ) \sim \mathcal { G P } \left( \mathcal { Q } ^ { \phi } ( \mathbf { x } ) m _ { 0 } ( \mathbf { x } ) , \mathcal { Q } ^ { \phi } ( \mathbf { x } ) k _ { 0 } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) \mathcal { Q } ^ { \phi T } ( \mathbf { x } ^ { \prime } ) \right) .\tag{8}
$$

As in PIGPs, the mean is frequently chosen to be zero. This construction requires one task per row of $Q ^ { \phi } ( \mathbf { x } )$ and one latent base kernel per column of $Q ^ { \phi } ( \mathbf { x } )$ . For q unknown functions, $Q ^ { \phi } ( \mathbf { x } )$ has at most $n \leq q$ columns and $q + 1 \leq p \leq 2 q$ rows, depending on how many components of $\vec { \xi } ( \mathbf { x } )$ are needed to model the inhomogeneities. So, this method never needs more tasks or latent functions than PIGPs, and typically needs fewer of both. This results in general in a stronger coupling between the remaining tasks which in turn can help improve performance on the inverse problem of learning physical parameters as we have shown in [25].

The NSB method [10] is applicable whenever a system is parametrizable. Otherwise, finding a $Q ^ { \phi }$ , such that $R ^ { \phi } Q ^ { \phi } = 0 .$ , only guarantees that every function generated through $Q ^ { \phi }$ is a solution of the differential equation. It does not yet guarantee the converse, namely that every solution of the differential equation can be generated through $Q ^ { \phi }$ . In other words, im( $\mathcal { Q } ^ { \phi } ) \subseteq \ker ( R ^ { \phi } )$ always holds, whereas parametrizability ensures the stronger equality im( $\mathcal { Q } ^ { \phi } ) = \ker ( R ^ { \phi } )$ . Therefore, for a non-parametrizable system, the parametrized GP is only able to realize certain solutions, but not all solutions, or can, in the (frequent) extreme case, only produce the trivial solution. We make this distinction precise in Sections 2.3–2.4.

## 2.3. A short introduction to some algebraic concepts

We will give a minimal introduction to necessary algebra in this section. Our work, however, remains focused on physics-informed machine learning, and not on algebra, so we intend to avoid overwhelmingly detailed definitions and provide just enough context to develop our method and for the reader to apply our results in a practical setting. For a detailed presentation of the rich topic of homological algebra, we refer interested readers for example to [31, 32, 33]. We have already introduced the class of rings containing our differential operators that we will work over as D in the last section. We keep the notation for the system matrix R as in Eq. 7, but drop parameter specifications φ and the argument (x) for better readability throughout the whole section.

Firstly, we need to introduce multidimensional objects over rings. Just like vector spaces can be spanned over fields, modules can be thought of as the analogue over rings D. Because the rings we consider are in general non-commutative, we distinguish between left and right modules, depending on which side elements act from. Because we have fixed commutation relations (such as Eq. 5 in the Weyl algebras), we can change between left and right modules easily via a mapping called involution (used frequently in computer algebra programs), so most arguments and definitions hold for both left and right modules, which is why we drop the specification at times.

Given a left module M, we define some characteristics:

Definition 2.1 ([31, 33]). We say

• M is stably free if there exists $r , s \in  { \mathbb { N } } _ { 0 } ,$ such that $M \oplus D ^ { 1 \times s } \cong D ^ { 1 \times \prime }$ , where ⊕ denotes the direct sum of modules.

• M is projective if there exists $r \in  { \mathbb { N } } _ { 0 }$ and a left D-module P such that M ⊕ $P \cong D ^ { 1 \times r }$

• M is torsion-free if the torsion left D-submodule defined by

$$
t ( M ) : = \{ m \in M | \exists d \in D \setminus \{ 0 \} : d m = 0 \}
$$

is reduced to $O , i . e . \ t ( M ) = \{ 0 \}$

• M is torsion $i f t ( M ) = M$

Similar definitions hold for right modules.

If M is stably free, it is automatically also projective, and if M is projective, it is also torsionfree [33]. If M is not torsion-free, we say it has torsion if $0 \neq t ( M ) \subset M$ , or is torsion, if $t ( M ) = M$

The following theorems hold for modules that are torsion:

Theorem 2.1 (Stafford's theorem [34] [29], adapted from Thm. 9, Cor. 2; cyclicity part adapted from [35, Thm. 4–5]). Let M be a finitely generated torsion left (resp. right) D-module, with D defined as in chapter 2.2.

(1) M can be generated by two elements.

(2) If, in addition, D is a ring of ordinary (i.e. single-variable) differential operators with polynomial, formal power series, or locally convergent power series coefficients, then M is cyclic, i.e. can be generated by a single element.

To study differential equations in operator space (i.e. over D-modules), we need to study the left quotient row-module of the differential equation characterized by the system matrix $R ,$ defined as

$$
M : = D ^ { 1 \times p } / ( D ^ { 1 \times q } R )\tag{9}
$$

i.e. $D ^ { 1 \times p }$ modulo the relations defined in the rows of R. In order to characterize $M ,$ we furthermore need to define its Auslander transpose $N : = D ^ { q } / ( R D ^ { p } )$ , and so called extension modules, written $\mathrm { e x t } _ { D } ^ { i } ( M , D )$ . These measure obstructions in chains of mappings from and to M, but only depend on the module M itself, not on the particular choice of the mapping. A more rigorous introduction based on algebraic diagrams can be found for example in [31]. For full row-rank matrices R ([35]), it holds that

$$
e x t _ { D } ^ { 1 } ( M , D ) \cong D ^ { q } / ( R D ^ { p } ) = N .\tag{10}
$$

Lemma 2.2 ([36, Prop. 2.2.1]). Let M be a finitely generated right (resp. left) D-module. Then $f o r i \geq 1$ the modules é $\mathbf { \nabla } : \mathbf { x t } _ { D } ^ { i } ( M , D )$ are either zero or finitely generated torsion left (resp. right) D-modules.

With M, N, and extension modules introduced, we can check characteristics of M in the following ways (for proofs we refer to cited literature):

Lemma 2.3 ([31]). Let $M = D ^ { 1 \times p } / ( D ^ { 1 \times q } R )$ be the left D-module finitely presented by the full row-rank matrix $R \in D ^ { q \times p }$ , and N its Auslander transpose. Then

1. M is torsion-free ⇔ $e x t _ { D } ^ { 1 } ( N , D ) = 0$

2. M is projective ⇔ $e x t _ { D } ^ { i } ( N , D ) = 0$ for all $i = 1 , 2 , 3 , 4 . .$

3. M is projective ⇔ R has a right inverse S, such that $R S = \mathbb { I } _ { q }$

4. M is stably free ⇔ $e x t _ { D } ^ { 1 } ( M , D ) = 0$

To study solutions of differential equations - i.e. functions - we need not only the operator characterized above through modules, but also a function space $\mathcal { F }$ over which we want to find these solutions. Formally, $\mathcal { F }$ is also a left D-module, and depending on the chosen ring D, $\mathcal { F }$ can be injective, or not. As a special case, it is well known that for a ring of differential ODE or PDE operators with constant coefficients, $C ^ { \infty }$ is an injective function space [37, 38]. However, for the more general case of D being a Weyl algebra or having general meromorphic coefficients, considered here, standard C function spaces including $C ^ { \infty }$ are not injective, which has consequences on parametrizability.

## 2.4. Algebraic parametrization

Combining the definitions introduced in the previous section, we can now study solutions ξ of our system (7). A graphical overview over the results of this section can be found in Fig. 2. The following relations hold

Lemma 2.4 ([31, Corollary 4.1]). Let $M = D ^ { 1 \times p } / ( D ^ { 1 \times q } R )$ be the left D-module fnitely presented by the full row-rank matrix $R \in D ^ { q \times p }$ . Let $\mathcal { F }$ be the function space over which we want to find solutions. Matrices always act on the right.

1. If M is torsion-free, there exists a non-trivial parametrization matrix $Q \in D ^ { p \times n }$ such that $R Q = 0 .$ Functions $\mathcal { F } ^ { p } \ni \vec { \xi } = Q \vec { f }$ with $\vec { f } \in \mathcal { F } ^ { n }$ are solutions of Eq. 7. Formally, we write imD $( Q ) = k e r _ { D } ( R )$ and imF $( Q ) \subseteq k e r _ { { \mathcal { F } } } ( R )$

2. If M is torsion-free and F is injective, all solutions in F can be parametrized using $Q ,$ i.e. $i m _ { \mathcal { F } } ( Q ) = k e r _ { \mathcal { F } } ( R )$

3. If M is projective, all solutions in $\mathcal { F }$ can be parametrized using $Q ,$ i.e. imF $( Q ) = k e r _ { \mathcal { F } } ( R )$ even $i f \mathcal { F }$ is not injective.

How many solutions can we parametrize?  
![](images/2a7cd2f711c2cb03f8f6ad3d64c7c3a52379c83ef1cca503e644005f2f6afbc7.jpg)  
Figure 2: Overview over parametrizability in operator space D and function space F. Every combination can be read as "Systems represented by modules in operator space that are [i.e. torsion-free] can parametrize [i.e. all] solutions over the function space F if [i.e. F is injective]"

4. If M is stably free, then M is in particular also projective (Definition 2.1), so (3) applies as well.

Definition 2.2 (Parametrizable systems). $I f i m _ { \mathcal { F } } ( Q ) = k e r _ { \mathcal { F } } ( R )$ , we call the system presented by R parametrizable over the function space F.

What happens if a system is not parametrizable over a certain function space? If a system over D is torsion-free, but not projective, and F is not injective (as is the case for the function spaces considered here), some solutions may not be realizable, though a large portion of solution space is. If a system on the other hand has torsion $( t ( M ) \subset M )$ , we can in general only parametrize some solutions (for example for specific initial or boundary conditions). As a special case, if a system is torsion $( t ( M ) = M )$ , the parametrization matrix Q is trivial $( Q = 0 )$ , and we can only parametrize the trivial, zero-in-every-task solution.

Remark: For linear systems governed by ordinary differential equations, the term parametrizability coincides with the notion of controllability in the sense of [39, 40] in linear systems- and control theory. In contrast, for systems governed by partial differential equations, parametrizability is a stronger property, i.e. parametrizability always implies controllability [41, 42] but not vice versa. Oftentimes in machine learning literature, parametrizability and controllability are used interchangeably [10, 11, 12].

Proposition 2.5 ([31, Definition 4.3, Theorem 4.2],[43, Theorem 8]). If a system presented by the full row-rank matrix $R \in D ^ { q \times p }$ is torsion-free, there exist minimal parametrizations $Q \in D ^ { p \times n }$ in the sense that the number of columns is minimal. This is the case, $i f n = \operatorname { r a n k } ( M ) = p - q .$

## 3. Ghost tasking

## 3.1. Bridging PIGP and NSB

With the notion of parametrizability established in the previous sections, we are now in a position to characterize the parametrized Gaussian Processes defined in 2.2. Lange-Hegermann [10] already observed that their method applies only to parametrizable systems, i.e., as established above, to projective systems, or at least torsion-free systems when working in an injective function space.

The method of Raissi et al. [9] for parametrizing a Gaussian Process $( \operatorname { E q } . 6 )$ , by contrast, was not originally introduced within an algebraic context. We show here that it can equally be interpreted as a parametrization, obtained by applying the matrix

$$
Q _ { P I G P } ( \mathbf { x } ) = \binom { \mathbb { I } _ { q } } { \mathcal { L } ^ { \phi } ( \mathbf { x } ) }
$$

to the base kernel $k _ { u u } ( \mathbf { x } , \mathbf { x } ^ { \prime } )$

Writing the system of Eq. 6 in homogeneous form,

$$
\begin{array} { r } { R _ { P I G P } ^ { \phi } ( \mathbf { x } ) \vec { \xi } _ { P I G P } ( \mathbf { x } ) = 0 , \quad R _ { P I G P } ^ { \phi } ( \mathbf { x } ) = \left( \mathcal { L } ^ { \phi } ( \mathbf { x } ) \quad - \mathbb { I } _ { q } \right) , \quad \vec { \xi } _ { P I G P } ( \mathbf { x } ) = \left( \vec { u } ( \mathbf { x } ) \right) , } \end{array}\tag{11}
$$

one sees that $Q _ { P I G P } ^ { \phi } ( \mathbf { x } )$ as defined above indeed satisfies

$$
\begin{array} { r l } { R _ { P I G P } ^ { \phi } ( \mathbf { x } ) Q _ { P I G P } ^ { \phi } ( \mathbf { x } ) = \left( \mathcal { L } ^ { \phi } ( \mathbf { x } ) } & { - \mathbb { I } _ { q } \right) \left( \underset { \mathcal { L } ^ { \phi } ( \mathbf { x } ) } { \mathbb { I } _ { q } } \right) = 0 . } \end{array}\tag{12}
$$

Moreover, we can show that $R _ { P I G P } ( \mathbf { x } )$ as defined in Eq. 11 has a right inverse, choosing

$$
S _ { P I G P } ^ { \phi } ( \mathbf { x } ) = \binom { S _ { 1 } } { \mathcal { L } ^ { \phi } ( \mathbf { x } ) S _ { 1 } - \mathbb { I } _ { q } }
$$

for any $S _ { 1 } \in D ^ { q \times q }$ , since

$$
\begin{array} { r l } { R _ { P I G P } ^ { \phi } ( \mathbf { x } ) S _ { P I G P } ^ { \phi } ( \mathbf { x } ) = \left( \mathcal { L } ^ { \phi } ( \mathbf { x } ) \right. } & { \left. - \mathbb { I } _ { q } \right) \left( \underset { \mathcal { L } ^ { \phi } ( \mathbf { x } ) S _ { 1 } - \mathbb { I } _ { q } } { S _ { 1 } } \right) = \mathbb { I } _ { q } . } \end{array}
$$

By point (3) of Lemma 2.3, the system presented by $R _ { P I G P } ^ { \phi } ( \mathbf { x } )$ is projective and hence parametrizable (Lemma $2 . 4 )$ over any (sufficiently smooth) function space.

Comparing $R _ { P I G P } ^ { \phi } ( \mathbf { x } )$ as defined in Eq. 11 with the original system matrix $R ^ { \phi } ( \mathbf { x } )$ in Eq. 7, we note that the latter is not necessarily projective, whereas $R _ { P I G P } ^ { \phi } ( \mathbf { x } )$ is projective by construction. The distinction between the two formulations lies entirely in how the inhomogeneities $f ( \mathbf { x } )$ are incorporated. In the traditional physics-informed GP formulation $R _ { P I G P } ^ { \phi } ( \mathbf { x } )$ , one inhomogeneity term is introduced per unknown function, even when $f ( \mathbf { x } ) = 0 { \mathrm { : } }$ so that $R _ { P I G P } ^ { \phi } ( \mathbf { x } )$ must always have 2q tasks and thus also 2q columns. In the formulation of Eq. 7 on the other hand, correlated inhomogeneities can be modeled using fewer tasks (so that $R ^ { \phi } ( \mathbf { x } )$ has $\textit { p } < 2 q$ columns), or, in the homogeneous case, omitted entirely, resulting in a system matrix $R ^ { \phi } ( \mathbf { x } )$ that has exactly $p = q$ columns. In this latter case, with zero inhomogeneities, the module presented by $R ^ { \phi } ( \mathbf { x } )$ is torsion (hence is neither torsion-free nor projective, see Def. 2.1), while the inclusion of q inhomogeneities resulting in $R _ { P I G P } ^ { \phi } ( \mathbf { x } )$ always guarantees projectivity. This raises a question of direct practical consequence: are q input tasks truly necessary, or can torsion-freeness or projectivity, and with it parametrizability, be achieved with fewer tasks? The answer is not merely of theoretical interest: it determines how much data and latent functions are actually required to obtain a well-posed, parametrized model. We show in the following that there exists a sweet spot reachable by our "ghost tasking" method, guaranteeing parametrizability while keeping tasks and latent functions at a minimum.

![](images/cbbdaa5d32858ff9aa39086a584b80f437f398464bc0a9b0321ce16bec97cffe.jpg)  
Idea: Lift system into higher dimension where it is parametrizable

![](images/00c58967f4e357b44f425c343b4e4870fddb571637af9eb2a1cd610da15c0645.jpg)  
Figure 3: Overview over the idea behind ghost tasking and its connection to physics-informed Gaussian Processes (PIGP) as in [9]. A non parametrizable system can be rendered parametrizable by adding extra tasks, and subsequently conditioning them away again. The differences between ghost tasking and PIGPs lie in (a) how many tasks we add and (b) how they are coupled to the system (Λ versus the identity matrix $\mathbb { I } _ { q } )$

## 3.2. Foundations of ghost tasking

Ghost tasking locates the optimum between the PIGP method [9] and the NSB method [10], because it provides a method for rendering a physical system encoded in a Gaussian Process minimally parametrizable. We propose a three-step procedure:

(i) introduce a maximum of 2 auxiliary tasks (i.e., auxiliary output variables) we call "ghost tasks" to embed the multi-dimensional GP in a higher-dimensional space, in which an otherwise unparametrizable system is reduced to a parametrizable system;

(ii) apply computer algebra methods to solve the resulting parametrizable system;

(iii) transform back to the original problem dimension by conditioning away the ghost tasks, i.e., conditioning them pointwise to vanish.

We visualize this idea and show the analogy to the previously discussed connection to PIGP in

Fig. 3. For better readability, we omit the parameters φ and the argument (x) at times. To formalize this procedure and prove its applicability, we use a theorem developed for Serre reduction:

Theorem 3.1 (Adapted from [35, Thm. 17], [44, Thm. 3.6]). Let D be a noetherian domain $R \in D ^ { q \times p }$ a full row-rank matrix, $\quad 0 \leq m \leq q , \Lambda \in D ^ { q \times m } , P = \left( R \quad - \Lambda \right)$ and $M = D ^ { 1 \times p } / ( D ^ { 1 \times q } R )$ (resp., $E = D ^ { 1 \times ( p + m ) } / ( D ^ { 1 \times q } P ) ) )$ the left D-module finitely presented by R (resp. P). Then, the following results are equivalent:

1. The left D-module E is stably free of rank $p - q + m .$

2. $\{ \tau ( \Lambda . _ { i } ) \} _ { i = 1 , \dots , m }$ generates the right D-module $e x t _ { D } ^ { 1 } ( M , D ) = D ^ { q } / ( R D ^ { p } )$ , where the right Dhomomorphism $\tau : D ^ { q } \to D ^ { q } / ( R D ^ { p } )$ is the canonical projection and Λ. is the $i ^ { t h }$ column of the matrix Λ.

Moreover, the construction depends only on the residue class of Λ i.e. $\overline { { { \Lambda } } } = \Lambda + R X$ yields the same result for any $X \in D ^ { p \times r }$

Proof sketch: For the full rigorous proof, we refer to [44]. From Lemma 2.3 we recall that E is stably free iff $\mathbf { e x t } _ { D } ^ { 1 } ( E , D ) = 0$ . Using Eq. 10 and the third isomorphism theorem ² applied to $R D ^ { p } \subseteq P D ^ { ( p + m ) } \subseteq { \bar { D ^ { q } } }$ , we rewrite

$$
\begin{array} { r } { \operatorname { e x t } _ { D } ^ { 1 } ( E , D ) \ \cong \ D ^ { q } / P D ^ { ( p + m ) } \ \cong \ ( D ^ { q } / R D ^ { p } ) / ( P D ^ { ( p + m ) } / R D ^ { p } ) \ \cong \ \operatorname { e x t } _ { D } ^ { 1 } ( M , D ) / ( P D ^ { ( p + m ) } / R D ^ { p } ) . } \end{array}
$$

Therefore,

$$
\operatorname { e x t } _ { D } ^ { 1 } ( E , D ) = 0 \quad \Longleftrightarrow \quad \operatorname { e x t } _ { D } ^ { 1 } ( M , D ) = P D ^ { ( p + m ) } / R D ^ { p } = ( R D ^ { p } + { \Lambda } D ^ { m } ) / R D ^ { p } .
$$

Applying the canonical projection $\tau : { \cal D } ^ { q }  { \cal D } ^ { q } / R { \cal D } ^ { p } = \mathrm { e x t } _ { { \cal D } } ^ { 1 } ( M , { \cal D } )$ , the right-hand side is simply the image of $\Lambda D ^ { m }$ under τ, i.e.

$$
( R D ^ { p } + \Lambda D ^ { m } ) / R D ^ { p } = \tau ( \Lambda D ^ { m } ) = \tau ( \Lambda ) D ^ { m } ,
$$

so that

$$
\mathrm { e x t } _ { D } ^ { 1 } ( E , D ) = 0 \quad \Longleftrightarrow \quad \mathrm { e x t } _ { D } ^ { 1 } ( M , D ) = \tau ( \Lambda ) D ^ { m } .
$$

That is, E is stably free exactly when the projections $\tau ( \Lambda . _ { i } )$ of the columns of Λ generate $\mathrm { e x t } _ { D } ^ { 1 } ( M , D )$ as a right D-module. □

Based on Theorem 3.1, we can now formalize ghost tasking for parametrized Gaussian Processes:

Theorem 3.2 (Ghost tasking). Let D denote a ring of linear differential operators whose coefficients are polynomial, rational, or general meromorphic functions. Let furthermore $R ^ { \phi } ( \mathbf { x } )$ ∈ $D ^ { q \times p }$ be a full row-rank matrix representing a linear system of ordinary or partial differential equations with coefficients in D, and M be the left module presented by R. Assume the system is not parametrizable.

Then, there exists a matrix $\Lambda ( { \bf x } ) \in { \cal D } ^ { q \times m }$ with at most m = 2 columns such that the embedded system $P ^ { \phi } ( \mathbf { x } ) = \left( R ^ { \phi } ( \mathbf { x } ) - \Lambda ( \mathbf { x } ) \right)$ is stably free, and hence parametrizable over any sufficiently smooth function space. The canonical projection of the columns of Λ generates $e x t _ { D } ^ { 1 } ( M , D ) \cong$

$D ^ { q } / ( R D ^ { p } )$ . Consequently, there exists a proper parametrization matrix $Q ^ { \phi } ( \mathbf { x } ) .$ such that every realization of the parametrized GP

$$
\eta \sim \mathcal { G P } ( 0 , Q ^ { \phi } ( \mathbf { x } ) k _ { 0 } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) Q ^ { \phi T } ( \mathbf { x } ^ { \prime } ) )
$$

solves the embedded system $P ^ { \phi } ( \mathbf { x } )$ . If we constrain the additional ghost task(s) to zero everywhere, the constrained solutions of the embedded system coincide with the solutions of the original system.

Proof. Assume R is not parametrizable, hence not projective. By Theorem 3.1, constructing a stably free module $E = D ^ { 1 \times ( p + m ) } / ( D ^ { 1 \times q } P ) , P = \big ( R \quad - \Lambda \big )$ , amounts to choosing $\Lambda \in D ^ { q \times m }$ whose projected columns generate $\mathrm { e x t } _ { D } ^ { 1 } ( M , D )$ ; the minimal such m is thus bounded by the minimal number of generators of $\mathrm { e x t } _ { D } ^ { 1 } ( \bar { M } , D )$ . By Lemma $2 . 2 , \mathrm { e x t } _ { D } ^ { 1 } ( M , D )$ is a finitely generated torsion module, and by (1) of Theorem 2.1 every such module is generated by at most two elements. Hence there exists $\Lambda \in D ^ { q \times m }$ with $m \le 2$ such that $E = D ^ { 1 \times ( p + m ) } / ( D ^ { 1 \times q } ( R , - \Lambda ) )$ is stably free, and therefore also projective.

For $_ \textit { D a }$ ring of ordinary differential operators, (2) of Theorem 2.1 strengthens this: every torsion module over $D$ is cyclic, so $\mathrm { e x t } _ { D } ^ { 1 } ( M , D )$ is generated by a single element and Λ may be chosen with $m = 1$

Since E is projective, $P \vec { \eta } = 0$ is parametrizable over any sufficiently smooth function space (Lemma 2.4), yielding a parametrized Gaussian Process with at most two ghost tasks $\xi _ { G T }$ . Writing

$$
\mathrm { O r i g i n a l : } R \vec { \xi } = 0 ,\tag{13}
$$

$$
\mathrm { N e w : ~ } P \vec { \eta } = \left( R \begin{array} { c c } { { } } & { { - \Lambda } } \end{array} \right) \left( \begin{array} { c } { { \vec { \xi } } } \\ { { \xi _ { G T } } } \end{array} \right) = R \vec { \xi } - \Lambda \xi _ { G T } = 0 ,\tag{14}
$$

with $\Lambda \neq 0$ (else R is already parametrizable), setting $\xi _ { G T } = 0$ recovers the original system, so the two are pointwise equivalent. □

## 3.3. Properties of ghost tasking

Having established the existence of a minimal ghost-tasking parametrization in Theorem 3.2, we now discuss several refinements, special cases, and practical considerations that follow from it.

Pointwise conditioning. In practice, a Gaussian Process cannot be constrained to vanish identically on a continuum, only at finitely many points. We therefore replace the pointwise condition $\xi _ { G T } ( \mathbf { x } ) = 0$ for all x with a finite set of conditioning points $\{ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { k } \}$ . Empirically, we find that a small to moderate number of conditioning points suffices in practice (Chapter 4).

Torsion-freeness. Theorem 3.2 guarantees the stronger characteristic of stable freeness, which implies torsion-freeness. For efficiency reasons (Chapter 4 and 5), it may suffice and be actually desirable to require only that the embedded module E be torsion-free, rather than stably free. Theorem 3.2 admits the following weaker, and thus more permissive, variant:

Proposition 3.3. Let $D , R , M , P ,$ and E be defined as in Theorem 3.2. There exists a matrix $\Lambda \in D ^ { q \times m }$ with at most $m = 2$ columns such that the embedded system is torsion-free. Writing $\tau ( \Lambda ) D ^ { m }$ for the submodule generated by the canonical projections of the columns of Λ, this holds precisely when

$$
\begin{array} { r } { \mathrm { e x t } _ { D } ^ { 1 } ( \mathrm { e x t } _ { D } ^ { 1 } ( M , D ) / ( \tau ( \Lambda ) D ^ { m } ) , D ) = 0 . } \end{array}
$$

This formulation subsumes the stably free case of Theorem 3.2: when τ $( \Lambda ) D ^ { m } \cong \operatorname { e x t } _ { D } ^ { 1 } ( M , D )$ $( \mathrm { i } . \mathrm { e } . , \Lambda ^ { \prime } ;$ s columns generate all of $\mathrm { e x t } _ { { D } } ^ { 1 } ( M , D ) )$ , the quotient $\mathrm { e x t } _ { D } ^ { 1 } ( M , D ) / \tau ( \Lambda ) D ^ { m }$ vanishes, and the condition holds trivially since ex $\mathrm { t } _ { D } ^ { 1 } ( \stackrel { \sim } { 0 } , D ) = 0$

Proof. We denote the Auslander transpose of $E$ by $N _ { E }$ . By Lemma 2.3, E is torsion-free, iff $\mathbf { e x t } _ { D } ^ { 1 } ( N _ { E } , D ) = 0$ , and since analogously to the proof of Theorem 3.1, we can write

$$
\begin{array} { r l r } & { } & { N _ { E } = D ^ { q } / P D ^ { ( p + m ) } \cong ( D ^ { q } / R D ^ { p } ) / ( P D ^ { ( p + m ) } / R D ^ { p } ) \cong \mathrm { e x t } _ { D } ^ { 1 } ( M , D ) / ( P D ^ { ( p + m ) } / R D ^ { p } ) } \\ & { } & { \cong \mathrm { e x t } _ { D } ^ { 1 } ( M , D ) / ( ( R D ^ { p } + \Lambda D ^ { m } ) / R D ^ { p } ) = \mathrm { e x t } _ { D } ^ { 1 } ( M , D ) / \left( \tau ( \Lambda ) D ^ { m } \right) , } \end{array}
$$

Therefore, iff $\mathrm { e x t } _ { D } ^ { 1 } ( \mathrm { e x t } _ { D } ^ { 1 } ( M , D ) / ( \tau ( \Lambda ) D ^ { m } ) , D ) = 0 .$ E is torsion-free. The number of columns is bounded by Theorem 3.2, and the rest of the proof follows the same arguments as Theorem 3.2.

Stronger conjecture. A long-standing conjecture of [34][Conjecture 3.8] asks, if any finitely generated torsion right module is cyclic. If this is true, the result would extend the single-column bound of to all Weyl algebras $A _ { n } ( k ) , n \geq 1$ , i.e. also to PDEs with polynomial coefficients. An AI-assisted machine proof in Lean [46] has recently been published [47], though has not been peer reviewed yet.

Recovering PIGP. Choosing $\Lambda = \mathbb { I } _ { q }$ recovers the PIGP construction of [9]: since $\tau ( \mathbb { I } _ { q } )$ trivially generates $\mathrm { e x t } _ { D } ^ { 1 } ( M , D )$ , this is always a valid — if not minimal — choice of $\Lambda ,$ corresponding to $m = q$ ghost tasks (i.e. one ghost task per equation).

Finding Λ in practice. Theorem 3.2 guarantees the existence of Λ but calculating a minimal set of generators is generally difficult. In the holonomic case, Λ can be computed directly using Macaulay2 [27]. Otherwise, we find the brute force method of systematically proposing Λ candidates containing low-order terms in D and verifying the validity of Λ computationally to be surprisingly effective. Verification is comparatively cheap in both the stably free/projective case (amounting to checking if extension modules are zero, Lemma 2.3), using a computer algebra system such as OreModules [26] or Macaulay2 [27]. We demonstrate the appropriate syntax in Appendix A.

Bound on the number of latent functions. Bounding the number of ghost tasks also bounds the number of latent functions needed to parametrize the solution space. By Proposition 2.5 the minimal parametrization matrix $Q$ of the embedded system $P \in D ^ { q \times ( p + m ) }$ has $n = ( p + m ) - q$ columns. Since $q \leq p$ (full row-rank) and $0 \leq m \leq q$ (with $m \le 2$ the minimal choice and $m = q$ recovering PIGP), n ranges between $p - q$ and $p - q + q = p$ latent functions. In the most extreme case of homogeneous differential equations $( R = \mathcal { L } ) , m \in \{ 1 , 2 \}$ columns (corresponding to one or two ghost tasks) suffice.

## 4. Experiments

We illustrate ghost tasking in three experiments, discussing different aspects of the implementation in each of them: the first experiment is an ODE, where we are learning the dynamics of a pendulum system. We show the advantage of knowing the ghost task to be exactly zero: conditioning the ghost task to be zero on additional collocation points allows us to better identify the eigenfrequency of the system. The second experiment is a PDE with two variables. In this case, the previously discussed notions of torsion-freeness and projectivity do not coincide anymore, and we compare the performance between the two parametrization variants. Lastly, we show the performance on a slightly more real-world example of a PDE with 3 variables in magnetostatics. For each experiment, we first derive the augmented parametrization matrix $Q ^ { \phi } ( \mathbf { x } )$ , and then apply the resulting minimally parametrized Gaussian Process obtained by ghost tasking (in short referred to as $" \mathrm { G T " } )$ to solve the forward and inverse problem. We implement the physicsinformed Gaussian Process (PIGP) method of [9] as a baseline comparison to our method in each experiment. For the second experiment, an explication of the commands used in the computer algebra programs OreModules or Macaulay2 can be found in Appendix A. Scripts and data for all experiments can be found on our GitHub³.

All numerical experiments use the python package PCGP [25], built on GPyTorch [48, 49] for optimization and jax [50] for grid evaluations. For each experiment we implement GT with the minimal number of ghost tasks, alongside PIGP as the comparison. In the forward problem, we only specify data of (i) the inhomogeneities, (ii) ghost task (or equivalent inhomogeneity tasks for PIGP), and (iii) boundary/initial conditions, and infer the solution on the unknown interior. Data points are noise-free, though we assume a Gaussian likelihood with small variance, $\sigma ^ { 2 } = 1 0 ^ { - 6 }$ , as a nugget for numerical stability. We choose the squared-exponential kernel as introduced in Sec. 2.2 as our base kernel and the calibration parameters $\vartheta = ( A , l )$ are optimized (ADAM, learning rate 0.1, 100 iterations) with a flat prior, A constrained to be positive and l constrained below by the minimal training-point distance. Accuracy is measured by RMSE on a test grid whose size depends on the experiment.

For the inverse problem, we systematically learn the maximum a posteriori (MAP) estimate of the physical parameters φ by optimization jointly with θ

$$
( \phi _ { M A P } , \vartheta _ { M A P } ) = \mathrm { a r g m a x } p ( \phi , \vartheta | \mathcal { D } , \sigma ) ,\tag{15}
$$

for varying numbers of data points D and noise levels σ. Each setting is repeated 10 times with different random noises to account for stochasticity. We define a run as converged if the physical parameters φ have less than 0.01% (relative to their mean) peak-to-peak variation within the last 50 iterations. Performance is measured by the MAP error relative to the true value of each physical parameter, $\frac { \phi _ { i , M A P } - \phi _ { i , t r u e } } { \phi _ { i , t r u e } }$

All computations were performed on a node of an x86\_64 Linux cluster (16 cores, 126 GB RAM) using 8 CPU cores per job.

## 4.1. Experiment 1: Tripendulum

The first experiment consists of a system of coupled ODEs.

Inspired by [11] and [25], we consider three pendula of length l that feel the gravitational acceleration g and are connected with a rod that moves with the acceleration $f ( t )$ . The linearized system can be written in the form $R ^ { ( \ell , g ) } ( { \bf x } ) \vec { \xi } ( { \bf x } ) = \vec { 0 }$ as follows:

$$
\left( { \begin{array} { c c c c } { \partial _ { t } ^ { 2 } + g / \ell } & { 0 } & { 0 } & { - 1 / \ell } \\ { 0 } & { \partial _ { t } ^ { 2 } + g / \ell } & { 0 } & { - 1 / \ell } \\ { 0 } & { 0 } & { \partial _ { t } ^ { 2 } + g / \ell } & { - 1 / \ell } \end{array} } \right) \left( { \begin{array} { c } { \theta _ { 1 } } \\ { \theta _ { 2 } } \\ { \theta _ { 3 } } \\ { f } \end{array} } \right) = \left( { 0 } \atop 0 \right) .\tag{16}
$$

If the lengths of the pendula are all the same, the system has torsion, and is therefore not parametrizable. Since the system consists of ODEs, the notion of torsion-freeness and projectivity coincides. To render the system parametrizable, we will therefore need to introduce a column vector Λ, such that its projection generates $D ^ { 3 } / R ^ { ( \ell , g ) } D ^ { 4 } \cong e x t _ { D } ^ { 1 } ( M , D )$ . In the ODE case, we can calculate this generator using Macaulay2 (see Appendix $\mathbf { A } )$ . Λ is not unique, in fact for any generator $\Lambda , \tilde { \Lambda } = \Lambda + R ^ { ( \ell , g ) } X$ for $X \in D ^ { 4 }$ are all equally valid choices. We use $\Lambda = ( - 1 , - \partial _ { t } , - t ) ^ { T }$ leading to the new system

$$
\left( { \partial } _ { t } ^ { 2 } + g / \ell \begin{array} { c c c c c } { { 0 } } & { { 0 } } & { { 0 } } & { { - 1 / \ell } } & { { 1 } } \\ { { 0 } } & { { \partial _ { t } ^ { 2 } + g / \ell } } & { { 0 } } & { { - 1 / \ell } } & { { \partial _ { t } } } \\ { { 0 } } & { { 0 } } & { { \partial _ { t } ^ { 2 } + g / \ell } } & { { - 1 / \ell } } & { { t } } \end{array} \right) \left( \begin{array} { c } { { \theta _ { 1 } } } \\ { { \theta _ { 2 } } } \\ { { \theta _ { 3 } } } \\ { { f } } \\ { { \xi _ { G T } } } \end{array} \right) = \left( { 0 } _ { 0 } ^ { 0 } \right) .\tag{17}
$$

that is parametrizable with the parametrization matrix

$$
Q _ { G T _ { 1 } } ^ { ( \ell , g ) } ( t ) = \left( \begin{array} { c c c } { { 1 } } & { { - \partial _ { t } ^ { 2 } \ell ^ { 2 } - g \ell } } \\ { { 1 } } & { { - \partial _ { t } ^ { 3 } \ell ^ { 2 } - g \ell \partial _ { t } } } \\ { { 1 } } & { { - \partial _ { t } ^ { 2 } \ell ^ { 2 } t + 2 \partial _ { t } \ell ^ { 2 } - g l t } } \\ { { \partial _ { t } ^ { 2 } \ell + g } } & { { 0 } } \\ { { 0 } } & { { \partial _ { t } ^ { 4 } \ell ^ { 2 } + 2 \partial _ { t } ^ { 2 } g \ell + g ^ { 2 } } } \end{array} \right) .\tag{18}
$$

The choice of using t in Λ means that we work in the non-commutative first Weyl algebra $D =$ $A _ { 1 } ( \mathbb { R } )$ . Since for ODEs, torsion-freeness and projectivity coincide, the system is parametrizable over $C ^ { \infty }$

For the inverse problem setting we will use $Q _ { G T _ { 1 } } ( t )$ to parametrize the GP kernel

$$
K ( t , t ^ { \prime } ) = \mathcal { Q } _ { G T _ { 1 } } ^ { ( \ell , g ) } ( t ) k _ { 0 } ^ { \vartheta } ( t , t ^ { \prime } ) \mathcal { Q } _ { G T } ^ { ( \ell , g ) T } ( t ^ { \prime } ) .
$$

For the forward problem, we would however typically use initial value conditions on $\theta _ { i } ( t = 0 )$ and on $\partial _ { t } \theta _ { i }$ for $i = { 1 , 2 , 3 }$ to uniquely identify the solution. To do so, we need to add additional tasks for $\partial _ { t } \theta _ { i }$ . Since $\partial _ { t } \theta _ { i }$ correlates with $\theta _ { i } ,$ we need to correctly couple the additional tasks to the parametrization of $\theta _ { i }$ . The parametrization of $\theta _ { i }$ corresponds to the operators of the i-th row of $Q _ { G T _ { 1 } } ^ { ( \ell , g ) } ( t ) \left( \mathrm { E q . } 1 8 \right)$ , so we parametrize $\partial _ { t } \theta _ { i }$ with $\partial _ { t }$ applied to the operators of the i-th row of $Q _ { G T _ { 1 } } ^ { ( \ell , g ) } ( t )$ Since we work in a Weyl algebra, we need to make sure to respect the commutation relation $[ \partial _ { t } , t ] = 1 ( \mathrm { E q . } 5 )$ when applying $\partial _ { t }$ . By adding the derivative tasks $\partial _ { t } \theta _ { i }$ to our parametrization matrix, we arrive at the parametrization matrix for the forward problem

$$
Q _ { G T _ { 2 } } ^ { ( \ell , g ) } ( t ) = \left( \begin{array} { c c c } { { 1 } } & { { - \partial _ { t } ^ { 2 } \ell ^ { 2 } - g \ell } } \\ { { 1 } } & { { - \partial _ { t } ^ { 3 } \ell ^ { 2 } - g \ell \partial _ { t } } } \\ { { 1 } } & { { - \partial _ { t } ^ { 2 } \ell ^ { 2 } t + 2 \partial _ { t } \ell ^ { 2 } - g \ell t } } \\ { { \partial _ { t } ^ { 2 } \ell + g } } & { { 0 } } \\ { { 0 } } & { { \partial _ { t } ^ { 4 } \ell ^ { 2 } + 2 \partial _ { t } ^ { 2 } g \ell + g ^ { 2 } } } \\ { { \partial _ { t } } } & { { - \partial _ { t } ^ { 3 } \ell ^ { 2 } - g \ell \partial _ { t } } } \\ { { \partial _ { t } } } & { { - \partial _ { t } ^ { 4 } \ell ^ { 2 } - g \ell \partial _ { t } ^ { 2 } } } \\ { { \partial _ { t } } } & { { - \partial _ { t } ^ { 2 } \ell ^ { 2 } - \partial _ { t } ^ { 3 } \ell ^ { 2 } t + 2 \partial _ { t } ^ { 2 } \ell ^ { 2 } - g \ell t } } \end{array} \right) .\tag{19}
$$

The baseline PIGP method interprets $f ( t )$ as three separate inhomogeneities of the form

$$
\mathcal { L } ^ { ( \ell , g ) } ( t ) \vec { \theta } ( t ) = \vec { f } ( t ) \Leftrightarrow \left( \begin{array} { c c c } { \ell \partial _ { t } ^ { 2 } + g } & { 0 } & { 0 } \\ { 0 } & { \ell \partial _ { t } ^ { 2 } + g } & { 0 } \\ { 0 } & { 0 } & { \ell \partial _ { t } ^ { 2 } + g } \end{array} \right) \left( \begin{array} { c } { \theta _ { 1 } } \\ { \theta _ { 2 } } \\ { \theta _ { 3 } } \end{array} \right) = \left( \begin{array} { c } { f _ { 1 } ( t ) } \\ { f _ { 2 } ( t ) } \\ { f _ { 3 } ( t ) } \end{array} \right) .\tag{20}
$$

Though we theoretically know $f _ { i } ( t ) = f ( t )$ describes the movement of the same rod in all three tasks, i.e. the tasks are perfectly correlated, we still need to model them as independent (inter-task correlation = 0) tasks to account for the different movements of the pendula.

This leads to the equivalent PIGP parametrization matrix (already including the derivative tasks $\partial _ { t } \theta )$

$$
\mathscr { Q } _ { P I G P } ^ { ( \ell , g ) } ( t ) = \left( \begin{array} { c } { \mathbb { I } _ { q } } \\ { 0 } \\ { \partial _ { t } \mathbb { I } _ { q } } \\ { \partial _ { t } \mathbb { I } _ { q } } \end{array} \right) = \left( \begin{array} { c c c c } { 1 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 1 } & { 0 } \\ { 0 } & { 0 } & { 1 } \\ { \ell \partial _ { t } ^ { 2 } + g } & { 0 } & { 0 } \\ { 0 } & { \ell \partial _ { t } ^ { 2 } + g } & { 0 } \\ { 0 } & { 0 } & { \ell \partial _ { t } ^ { 2 } + g } \\ { \ell \partial _ { t } } & { 0 } & { 0 } \\ { 0 } & { \ell \partial _ { t } } & { 0 } \\ { 0 } & { 0 } & { \ell \partial _ { t } } \end{array} \right) .\tag{21}
$$

For the numerical experiments, we can directly use $Q _ { G T } ^ { ( \ell , g ) } ( t )$ resp. $Q _ { P I G P } ^ { ( \ell , g ) } ( t )$ as input for the package PCGP [25] that automatically generates the corresponding kernels that can then be used either with GPyTorch [48, 49] or jax [50].

The ground truth solution of all three angles $\theta = \theta _ { i } , i = 1 , 2 , 3$ given fixed initial conditions $\theta ( 0 ) = \theta _ { 0 } , \dot { \theta } ( 0 ) = \dot { \theta } _ { 0 }$ and a chosen forcing acceleration $f ( t ) = 5 \sin ( \omega _ { f } t )$ is

$$
\theta ( t ) = \theta _ { 0 } \cos ( \omega t ) + \left( \dot { \theta } _ { 0 } - \frac { 5 \omega _ { f } } { \ell ( \omega ^ { 2 } - \omega _ { f } ^ { 2 } ) } \right) \frac { 1 } { \omega } \sin ( \omega t ) + \frac { 5 } { \ell ( \omega ^ { 2 } - \omega _ { f } ^ { 2 } ) } \sin ( \omega _ { f } t ) , \qquad \omega = \sqrt { \frac { g ^ { 2 } } { \ell } } ,\tag{22}
$$

with chosen parameter values $g = 9 . 8 1 , \ell = 2 . 5 , \omega _ { f } = 1 . 6$ The training data for the forward problem consists of a total of six initial condition points at $\theta _ { i } ( t = 0 )$ and $\partial _ { t } \theta _ { i } ( t = 0 ) , \ i = 1 , 2 , 3$ and 10 linearly spaced points for $f ( t )$ in GT and the same 10 points in all $f _ { i } ( t ) , \ i = 1 , 2 , 3$ in the PIGP method. We compare two different datasets for the ghost task: because we know $\xi _ { G T } ( t ) = 0$ everywhere without needing to measure it, we can use as many collocation points to enforce this pointwise as we want. We compare 10 to 20 linearly spaced collocation points and refer to the version with additional linearly spaced collocation points as $" \mathrm { G T } + "$ . The RMSE was calculated over 201 linearly spaced test points in each task. For the inverse problem we are comparing the MAP estimates of the parameters l and $g ,$ as well as their ratio $g / \ell$ for different amounts of training data per task $n = 3 , 5 , 1 0$ and different noise levels $\sigma \in \{ 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 0 . 2 \}$ . While the ratio $g / \ell$ we compare is not an independent parameter, Eq. 22 shows that $\begin{array} { r } { \omega = \sqrt { \frac { g } { \ell } } } \end{array}$ dominates the dynamics of the system and is therefore an important quantity to assess.

As in the forward problem, we compare a "standard" GT method where we condition the ghost task on n collocation points, i.e. equally many points as the other tasks, and $\mathrm { ~ a ~ } \ " \mathrm { G T } + \ " $ method, where we condition the ghost task on 25 collocation points to better constrain the ghost task.

Since we have two parameters of interest here, it is also interesting to look at the joint posterior probability of l and g (Eq. 23), that is highly non-trivial. We calculate it on a $2 0 { \times } 2 0 { \times } 2 0 { \times } 2 0$ linearly spaced point grid over the parameters $\ell , g , A , l$ and subsequently marginalize over the calibration parameters amplitude A and lengthscale l. We assume a flat prior over the considered ranges g ∈ [1, 15], l ∈ [0.5, 5], A ∈ [0.5, 100], $l \in [ l _ { m i n } , l _ { m a x } ]$ with $l _ { m i n } , l _ { m a x }$ being the closest and farthest distance between training points.

$$
p ( \ell , g | \mathcal { D } ) = \int p ( \ell , g , A , l | \mathcal { D } ) d A d l \approx \frac { 1 } { Z } \sum _ { i = 1 } ^ { 2 0 } \sum _ { j = 1 } ^ { 2 0 } p ( \ell , g , A _ { i } , l _ { j } )\tag{23}
$$

We will qualitatively assess the performance of GT, GT+, and PIGP compared to the optimal posterior resulting from a full Bayesian parameter estimation using a Gaussian likelihood centered around the true forward solution (Eq. 22) as a function of $g$ and $\ell .$

In Fig. 4 we show the forward solution of each task, on the left side for GT, in the middle for $\mathrm { G T + }$ , and on the right side for PIGP. The RMSE of the mean solution calculated over the test points is written in the corner of each task and ranges between 0.0001 and 0.1. We see that overall, the RMSE of the GT method using a smaller amount of total datapoints than GT+ and PIGP are similar to PIGP, though the uncertainty is high in some places. The additional collocation points of GT+ reduce this uncertainty. In total, GT+ uses the same amount of training data as PIGP, but the RMSE for GT+ is lower than for PIGP.

We observe in Fig. 5 that GT achieves a comparable, and in $g / \ell$ about one order of magnitude improved performance compared to PIGP in the low data regime. However, GT+ improves especially the $g / \ell$ results up to $^ { 5 }$ orders of magnitude. We believe the performance of ghost tasking may be attributed to the inter-coupling between the tasks corresponding to the angles $\theta _ { i } ( t )$ , which allows the system to share information on the joint dynamics [25]. Constraining on additional collocation points in the ghost task forces the fitting error of the ghost task, that can be seen as a residue, closer to zero, progressively suppressing oscillatory components in the ghost task. Therefore, no dynamically important oscillations can be absorbed in the ghost task as a residue, considerably improving the learning of the frequency-related ratio. The additional collocation points cannot be provided to PIGP in this experiment, since the forcing term $f ( t )$ is treated as an unknown quantity that must itself be inferred from measured data. PIGP recovers the performance of GT in the case where $f ( t )$ is assumed to be fully known, as we show in the ablation studies in Tab. B.1.

The full posterior distributions depicted in Fig. 6 corroborate this analysis. The figure displays the transformed log posterior landscapes4 in the (g, l)-plane for all combinations of n and σ, comparing the reference full Bayesian estimate to GT, GT+ and PIGP. The elongated structures in the posterior indicate a strong correlation between g and l that is explained by the system dynamics depending primarily on $\omega = \sqrt { g / \ell }$ Furthermore, the reference standard Bayesian posterior exhibits multimodality, with several distinct frequency modes corresponding to solutions with different frequencies passing through the same training points. Although the true frequency mode has substantially higher posterior probability, the presence of additional modes reflects the inherent ambiguity of parameter inference from oscillatory observations. Importantly, the same multimodal structure is recovered by $\mathrm { G T + }$ , while multimodality is present in GT, but with a slightly shifted frequency, pointing to physical oscillations being absorbed in the ghost task as previously discussed. In general, this indicates that the ghost tasking method captures this aspect of the underlying posterior geometry. In contrast, the PIGP posterior is unimodal and is not able to reproduce these additional physically meaningful modes in the considered setting, which can lead to biased posteriors as can be seen for $n = 3$ . When the forcing term $f ( t )$ is assumed to be known (almost) exactly, PIGP is also able to recover the multimodal posterior structure, as shown in our ablation studies in Fig. B.14 in Appendix B.

![](images/19c6efa9a9efa3f104d8e0403d9e20c2f752e56ebac92e2055f1157d38b10912.jpg)  
Figure 4: Inferred solutions of the forward problem of experiment 1 for the GT method (left), GT+ with extra collocation points (middle), and the PIGP method (right). Each subplot shows the solution of one task, the last three are the derivative of the angles in the first three tasks, necessary to constrain a second order initial condition problem. The true solution is marked in red, the training data with black stars, and the inferred solutions in blue with a lightblue 2σ uncertainty band. The RMSE on the test points is written in the corner of each task

![](images/7ab160f7c874a1c256714b43ae81dcac9d9f1bba36fd83c6593182d7e5d1b5c5.jpg)  
Figure 5: Results of the systematic comparison in experiment 1 of GT (green), GT+ (orange), and PIGP (blue) using the mean absolute error of the learned physical parameters (maximum a posteriori, MAP value) l (first row), g (middle $\operatorname { r o w } ) _ { \mathrm { { ; } } }$ as well as their ratio $g / \ell$ (last row) relative to their true values $( \ell , g ) = ( 2 . 5 , 9 . 8 1 )$ as metric. Note that $g / \ell$ is not an independent parameter, but does dictate much of the dynamics of the system and can as such be most precisely estimated by $\mathrm { G T + . }$ The experiments have been conducted for different numbers of training data $n \in \{ 3 , 5 , 1 0 \}$ for the tasks, as well as different noise levels $\sigma \in \{ 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 0 . 2 \}$ , shown in different subplots. We show the distinct runs as scatterplots, as well as their box-plots for each combination (σ, n).

![](images/3ede9eed748970375f2f6ba552665eb31429f2c771bc6c028e1f404578fe7014.jpg)  
Figure 6: Re-scaled log posteriors − log(max(log(p(l, g))) − log(p(l, g)) + 1) in the $( g , \ell )$ -plane marginalized over the calibration parameters A and l for all combinations of n and σ. The reference Bayesian estimate for the known solution (upper left plot) is compared to GT (upper right), GT+ (lower right), and PIGP (lower left). The elongated structures show a strong correlation between g and l, and we see multimodality in the reference and GT posteriors. We do not show colorbars because the likelihood scales vary substantially between panels; the comparison is qualitative and focuses on the relative structure of each likelihood rather than absolute values compared across panels.

## 4.2. Experiment 2: Coupled transport

We will now assess the performance of ghost tasking on a system of PDEs with two input variables, shown in Eq. 24 that includes all possible cases and complications, but is still simple enough to be solved analytically as well.

$$
\begin{array} { r } { \partial _ { t } u + x \partial _ { x } u + \alpha \partial _ { x } \nu = 0 } \\ { \partial _ { t } \nu + x \partial _ { x } \nu = 0 } \end{array}\tag{24}
$$

A physical interpretation of this system could be a coupled transport in a shear flow field with some coupling constant α for example. Note that the coefficients are polynomials in x, so the second Weyl Algebra (over $\mathbf { x } = ( x , t ) )$ is a suitable choice of operator ring $D .$ We define our operator matrix $R ^ { \alpha } ( \mathbf { x } )$ by rewriting Eq. 24

$$
\begin{array} { r } { R ^ { \alpha } ( \mathbf { x } ) \vec { \xi } ( \mathbf { x } ) = \left( \begin{array} { c c } { \partial _ { t } + x \partial _ { x } } & { \alpha \partial _ { x } } \\ { 0 } & { \partial _ { t } + x \partial _ { x } } \end{array} \right) \left( \begin{array} { c } { u ( x , t ) } \\ { \nu ( x , t ) } \end{array} \right) = \vec { 0 } . } \end{array}\tag{25}
$$

Alas, the original system is torsion and is therefore not parametrizable. We will try to render the system parametrizable by introducing a ghost task and a corresponding column vector Λ(x). We have two options going forward: Using theorem 3.2, we can construct an augmented system that is projective (in short $" \mathrm { p r o j " } )$ , i.e. guaranteed to be able to realize all solutions in any sufficiently smooth function space, by choosing $\Lambda ( \mathbf { x } )$ , and the corresponding module $\tau ( \Lambda ) D ^ { m }$ its projection generates, such that $e x t _ { D } ^ { 1 } ( M , D ) / ( \tau ( \Lambda ) D ^ { m } ) = 0 , \mathrm { i . e }$ $\tau ( \Lambda ) D ^ { m } \cong e x t _ { D } ^ { 1 } ( M , D ) .$ On the other hand, the condition $e x t _ { D } ^ { 1 } ( e x t _ { D } ^ { 1 } ( M , D ) / ( \tau ( \Lambda ) D ^ { m } ) , D ) \ = \ 0$ of Proposition 3.3 to render the augmented system torsion-free (in short "tf") is weaker, and finding a valid Λ(x) in this sense by brute force search can be significantly faster, and heuristically results in shorter, lower order term structures in the resulting parametrization matrix $Q ^ { \alpha } ( \mathbf { x } )$ . On the downside, since we work over the second Weyl algebra, torsion-freeness does not guarantee that our parametrization captures all solutions over $C ^ { \infty }$ (see chapter 2.4). Although this may be mathematically unsatisfying, from an engineering and efficiency standpoint we nonetheless recommend using the torsion-free version whenever it is applicable, as we demonstrate in this experiment and the next. For this experiment, we will explore and compare both versions. For the projective system, a suitable $\Lambda ( \mathbf { x } )$ is $\Lambda _ { p r o j } ( { \bf x } ) = ( 1 , t )$ , resulting in the parametrization matrix

$$
Q _ { \mathrm { G T } \geq \infty } ^ { \infty } ( \mathbf x ) : = ( \begin{array} { c c c } { \alpha ^ { 1 2 } x \partial _ { x } ^ { 2 } + \alpha t ^ { 2 } \partial _ { t } \partial _ { x } - 2 \alpha t \partial _ { x } ^ { 2 } - t x ^ { 2 } \partial _ { x } ^ { 2 } - 2 \alpha t \partial _ { t } \partial _ { x } + \alpha t ^ { 2 } \partial _ { x } - 2 t x \partial _ { x } \partial _ { t } + 2 x ^ { 2 } \partial _ { x } ^ { 2 } - 2 \alpha t \partial _ { x } - t \partial _ { t } ^ { 2 } + 4 x \partial _ { x } \partial _ { t } + 4 \alpha \partial _ { x } + 2 \partial _ { t } ^ { 2 } + t \partial _ { t } - 3 x \partial _ { x } - 5 \partial _ { t } + 1 } \\ { - t x ^ { 2 } \partial _ { x } ^ { 2 } - 2 t x ^ { 2 } \partial _ { x } ^ { 2 } - 2 t x \partial _ { t } \partial _ { x } + 2 t x ^ { 2 } \partial _ { x } ^ { 2 } - t \partial _ { t } ^ { 2 } + 4 t \partial _ { x } \partial _ { t } + 2 t \partial _ { t } ^ { 2 } + t \partial _ { t } ^ { 2 } - 2 t x \partial _ { x } - 4 t \partial _ { t } - 2 x \partial _ { x } - 2 \partial _ { t } + 4 } \\ { - t x ^ { 3 } \partial _ { x } ^ { 3 } - 3 t x ^ { 2 } \partial _ { t } \partial _ { x } ^ { 2 } + 2 x ^ { 3 } \partial _ { x } ^ { 3 } - 3 t x \partial _ { t } ^ { 2 } \partial _ { x } + 6 x ^ { 2 } \partial _ { t } ^ { 2 } \partial _ { x } ^ { 2 } - 2 t x ^ { 2 } \partial _ { x } ^ { 2 } - t \partial _ { t } ^ { 3 } + 6 x \partial _ { x } \partial _ { t } ^ { 2 } - t x \partial _ { x } \partial _ { t } + 2 \partial _ { t } ^ { 3 } + t \partial _ { t } ^ { 2 } - 6 x \partial _ { x } \partial _ { t } - 6 \partial _ { t } ^ { 2 } - 2 x \partial _ { x } + 2 \partial _ { t } } \end{array}\tag{26}
$$

On the other hand, choosing the $\Lambda _ { t f } = ( 1 , \partial _ { t } )$ renders the system torsion-free and leads to the parametrization matrix

$$
Q _ { G T t f } ^ { \alpha } ( \mathbf { x } ) = \left( \begin{array} { c } { - \alpha \partial _ { t } \partial _ { x } + x \partial _ { x } + \partial _ { t } - 1 } \\ { \partial _ { t } \partial _ { x } x + \partial _ { t } ^ { 2 } - \partial _ { t } } \\ { \partial _ { x } ^ { 2 } x ^ { 2 } + 2 \partial _ { t } \partial _ { x } x + \partial _ { t } ^ { 2 } - \partial _ { t } } \end{array} \right) .\tag{27}
$$

The corresponding (projective) PIGP parametrization, equivalent to choosing $\Lambda = { \binom { 1 } { 0 } } \quad 0 \rangle$ , is

$$
Q _ { P I G P } ^ { \alpha } ( \mathbf { x } ) = \left( \begin{array} { c c } { 1 } & { 0 } \\ { 0 } & { 1 } \\ { \partial _ { t } + x \partial _ { x } } & { \alpha \partial _ { x } } \\ { 0 } & { \partial _ { t } + x \partial _ { x } } \end{array} \right) .\tag{28}
$$

Comparing Eq. 26 and $\operatorname { E q . 2 7 }$ to Eq. 28, we note that both $Q _ { G T p r o j } ^ { \alpha } ( \mathbf { x } )$ and $Q _ { G T t f } ^ { \alpha } ( \mathbf { x } )$ have three rows, and therefore needs three tasks, while $Q _ { P I G P } ^ { \alpha } ( \mathbf { x } )$ needs four tasks. Furthermore, $Q _ { G T p r o j } ^ { \alpha } ( \mathbf { x } )$ and $Q _ { G T t f } ^ { \alpha } ( \mathbf { x } )$ have one column, meaning that the latent space consists of one function, while $Q _ { P I G P } ^ { \alpha } ( \mathbf { x } )$ has two columns and therefore needs two independent functions spanning the latent space in order to express all solutions of our system.

To generate training data, we use the analytical solution obtained by the method of characteristics,

$$
\begin{array} { r l } & { u ( t , x ) = u _ { 0 } ( x \exp ( - t ) ) - \alpha ( 1 - \exp ( - t ) ) \nu _ { 0 } ^ { \prime } ( x \exp ( - t ) ) , } \\ & { \nu ( t , x ) = \nu _ { 0 } ( x \exp ( - t ) ) . } \end{array}\tag{29}
$$

Providing initial conditions $u ( x , 0 ) = 0$ and $\nu ( x , 0 ) = 3 \sin ( 2 x )$ and choosing the true parameter to be $\alpha = 2$ renders the solution unique. For the forward problem, both GT versions require conditioning on the full grid in 1 ghost task, while PIGP requires conditioning on the full grid in 2 tasks. We provide initial conditions in both cases. The domain of interest is $\Omega = \{ ( x , t ) ^ { T }$ $x , t \in [ 0 , 1 ] \}$ , but we place collocation points on a wider region: the fully conditioned tasks are conditioned on a $1 1 \times 1 0$ grid over $x \in [ - 0 . 1 , 1 ] , t \in [ 0 , 1 ]$ , and the initial conditions of u and v are enforced on 11 linearly spaced points $( x , 0 )$ for $x \in [ - 0 . 1 , 1 ]$ . This extension outside Ω is possible because the parametrization is valid on all of $\mathbb { R } ^ { 2 }$ , and we use it for stability reasons: In this example, the characteristics transport information along $x = x _ { 0 } e ^ { t }$ , so every value near the corner $( x , t ) = ( 0 , 1 )$ traces back to a small neighborhood of $( 0 , 0 )$ , making that region especially sensitive to variations near (0, 0). By also constraining on $x \in ( - 0 . 1 , 0 )$ , we make sure that the initial condition is well represented near $( 0 , 0 )$ . In total, this means we condition on $1 { \cdot } 1 1 0 { + } 2 { \cdot } 1 1 =$ 132 training points for our ghost tasking methods and $2 \cdot 1 1 0 + 2 \cdot 1 1 = 2 4 2$ training points for the PIGP method. This data inequality stems from the different amount of tasks we need to condition on and ensures the same data density is provided to both methods. The test points lie on a linear $5 1 \times 5 1$ grid for each task, and only points that lie in the domain Ω are considered for the RMSE calculations.

For the inverse problem, analogously to experiment 1, we systematically generated training points on $n \times n$ grids with $n = 3 , 5 $ , 10 for all tasks. In our ablation studies Appendix B, we also considered the previously introduced $" \mathrm { G T } + "$ method using additional collocation points. Concretely, the collocation grid for the ghost task is augmented to a $1 0 \times 1 0$ instead of ${ \mathrm { ~ a ~ } } n \times n$ grid. Because we know the tasks corresponding to inhomogeneities in the PIGP method also to be exactly zero, we can also consider an augmented $1 0 \times 1 0$ collocation grid for the two inhomogeneity tasks in the PIGP method. We also study this augmented method in our ablation studies in Appendix B and analogously call this variant $" \mathrm { P I G P } { + } "$ . While the additional collocation points improved the results slightly in both methods, we did not find the same level of improvement relative to the increased computation time as in experiment 1, so we only discuss the standard cases "GT" and $" \mathrm { P I G P " }$ , with the same amount of training points in each task, in the main text.

Fig. 7 shows an example solution of the forward problem given the ghost tasks and the initial conditions for both the GT (upper two rows) and the PIGP method (lower two rows). We see that both methods perform comparably well with an RMSE below 0.006 on every task. Overall, the projective version of GT outperforms the other two slightly, while using less training data than PIGP. It shall be noted however, that inference for projective GT took about 25 seconds, while for torsion-free GT and PIGP, it took about 3 seconds

The results of the inverse problem are shown in Fig. 8. All methods perform comparably across settings. This matches our expectations, since the underlying model is only loosely coupled (with $\mathcal { L } ^ { \alpha } ( x , t )$ being a triangular matrix) – a setting in which the loosely-coupled PIGP approach is already well suited. We especially note that the performances of the torsion-free and the projective GT versions are similar, however, since the parametrization matrix for projective GT (eq. 26) contains more terms, the kernel evaluation takes much more time, resulting in convergence times that were approximately 10 times longer than for the torsion-free GT and the PIGP case. In summary, both torsion-free ghost tasking and projective ghost tasking recover the performance of PIGP in this experiment while using less training data.

## 4.3. Experiment 3: Anisotropic Magnetostatics

Next we discuss a magnetostatics example as in [51]: For a simply connected domain Ω, we can write the magnetostatic equations for the magnetic flux density $\vec { B } ( x , y , z )$ in terms of a vector

![](images/1c15572fb2b92d5c9e1bcfdd18dd1eb46f282e72fe9f64006400b7f99f4f2f14.jpg)

![](images/e7278ee5f05804db74ec3771156f1d84fed85675d16e907957d4ecf8b9e6c720.jpg)

![](images/4e81a1959c95e6f777e44ce4c3f5331521f769bf4d6ef8ce646bb97db1e00271.jpg)

![](images/6e1736fbae625addb545437068f036ba497a04dc3997de0dcefd679adbc9b006.jpg)

![](images/1a54b5d38b50ca9d980f93d4b99af6c7cc1dd070c1d201a473250e77b9c0d66d.jpg)

![](images/f8ac2593619151f53f68a0f8f7d16826af835e55a8b5d9a527f64507ca8a18ab.jpg)

![](images/dafdd9222d458470b677abaa1c747c0334bd06a70cd07ca9e15349c77024ffa7.jpg)

![](images/61f958afb557e980cdea480be90d09fcffc191d9c0f4767a50d99941449f10d6.jpg)

![](images/de2cebb2de6abbf42b62043581de2054a37b924be2f9489ab4bc7e1d820e49cb.jpg)

![](images/2e1a32b5586a6b37aa0f904989b9e4b815d4c5b1a2e465cfd4ca6ad5ef01f8ca.jpg)

![](images/90f055bb8dd5f3ec57832911a458baa48763475ca957ca495e66d6c35104eb28.jpg)

![](images/ee3328e967130dad0951b645007d504d8aaaaacd2c7e56671e57f941b10bbc28.jpg)

![](images/193fc354c5b00ba4aac60bf33b06befb789f35aba1163f7eb64b5738db3849bb.jpg)

![](images/e5bce3ef29c6b42a1d4ce68b0bf48670d0e92e418118414bb8a88dc404c99c49.jpg)

![](images/856b7be3cad7c478c34fcad56a564dd169567203bbc7c01dbf252dbad9fdd000.jpg)

![](images/f316d42025a3b4184d55c223aff5c94ad3f60235e3295171ed9e6319da46b6f2.jpg)

![](images/ad20045113636a97592c299fb5e592cfbd98d516c0db0a4ebb5439cdf08fb501.jpg)

![](images/747d8bf11c8b3018708e3ceea2369226861634f6ce7b021b1125b46b901e674a.jpg)

![](images/a37c40af71219d26eab292cc7ab86f78c020d41fa874f3c8f931ad67c0a4d50f.jpg)

![](images/4cb7a59254b30197f0dda2a5485731bbf59b7e90462eb63037c03f682f71446c.jpg)  
Figure 7: Inferred solution (uneven rows) and absolute error (even rows) of the forward problem given training data for the ghost task/input tasks and the initial conditions. The torsion-free GT version is depicted in the first two rows, the projective GT version in the middle two rows, and PIGP in the lowest two rows. The left column shows the results for u, the middle column for v and on the right we show the ghost task resp. input tasks for the inhomogeneity necessary. Conditioned points are marked with blue. The RMSE is computed within the region of interest, marked by a red box.

potential $\vec { A } ( r )$ with $\nabla \times \vec { A } = \vec { B }$ through the curl-curl equation with Dirichlet boundary conditions

$$
\nabla \times ( \nu \nabla \times { \vec { A } } ) = { \vec { J } } ,\tag{30}
$$

$$
{ \vec { n } } \times { \vec { A } } = 0
$$

on ∂Ω

(31)

![](images/415801909c08465cd97a04260d3246b2cea24784107f1ca61c33aceae5c5d6a6.jpg)  
Figure 8: Relative error as performance metric of the inverse problem of experiment $^ { 2 , }$ analogously to Fig. 5 of experiment 1. The results for the torsion-free GT system are shown in green, the results for the projective GT version are shown in orange and the results for PIGP are shown in blue.

$\nu ( x , y , z )$ refers to the reluctivity tensor, i.e. the inverse of the permeability, and $\vec { J ( } x , y , z { ) }$ is the current density. We can write the curl operator $\nabla \times$ explicitly as an operator matrix

$$
( \nabla \times ) = \left( \begin{array} { c c c } { { 0 } } & { { - \partial _ { z } } } & { { \partial _ { y } } } \\ { { \partial _ { z } } } & { { 0 } } & { { - \partial _ { x } } } \\ { { - \partial _ { y } } } & { { \partial _ { x } } } & { { 0 } } \end{array} \right) .\tag{32}
$$

$$
0 \Big ) ^ { T }
$$

For this example, we choose $\vec { J ( } x , y , z { ) }$ to be aligned with the x-axis, i.e. $\vec { J } ( x , y , z ) = \left( J _ { x } ( x , y , z ) \right.$ We consider an inhomogeneous material in a box $\Omega = \lbrace ( x , y , z ) ^ { T } : x , y \in ( 0 , 1 ) , z \in ( 1 , 2 ) \rbrace$ with a local reluctivity $\nu ( x , y , z ) = \mathrm { d i a g } ( \nu _ { 0 } , \nu _ { 0 } , \nu _ { 0 } z )$ . It behaves symmetrically in the $x - y$ plane but its reluctivity increases in z direction with its position. This could be $\mathrm { e . g }$ . an engineered metamaterial, or an approximation for a layered composite material. Then, we can formulate our problem in terms of operator matrices

$$
\begin{array} { r } { ( \nabla \times ) \nu ( \nabla \times ) \vec { A } = \vec { J } \Leftrightarrow \left( \begin{array} { c c c c } { - \partial _ { y } ^ { 2 } z \nu _ { 0 } - \partial _ { z } ^ { 2 } \nu _ { 0 } } & { \partial _ { x } \nu _ { 0 } \partial _ { y } z } & { \partial _ { z } \nu _ { 0 } \partial _ { x } } & { 0 } \\ { \partial _ { x } \nu _ { 0 } \partial _ { y } \nu _ { 0 } z } & { - \partial _ { x } ^ { 2 } z - \partial _ { z } ^ { 2 } \nu _ { 0 } } & { \partial _ { z } \nu _ { 0 } \partial _ { y } } & { 0 } \\ { \partial _ { z } \nu _ { 0 } \partial _ { x } } & { \partial _ { z } \nu _ { 0 } \partial _ { y } } & { - \partial _ { x } ^ { 2 } \nu _ { 0 } - \partial _ { y } ^ { 2 } \nu _ { 0 } } & { - 1 } \end{array} \right) \left( \begin{array} { c } { A _ { 1 } } \\ { A _ { 2 } } \\ { A _ { 3 } } \\ { J _ { x } } \end{array} \right) = \vec { 0 } . } \end{array}\tag{33}
$$

This system is not parametrizable. Since it is a system of PDEs, we can distinguish between a torsion-free and a projective augmented system. We were able to calculate a projective $\Lambda _ { p r o j } =$ $( 0 , 1 , x z )$ . However, calculating the corresponding parametrization matrix was not feasible on a laptop in a reasonable time. Instead, we are opting to build a torsion-free system by adding the column vector $\Lambda = ( 0 , 1 , 0 ) ^ { T }$ , and show that already the torsion-free system is indeed sufficient to vastly outperform PIGP in this experiment. The torsion-free parametrization matrix is

$$
\begin{array} { r } { Q _ { G T } ^ { \nu _ { 0 } } ( \mathbf { x } ) = \left( \begin{array} { c c } { - \partial _ { x } } & { - \partial _ { y } } \\ { - \partial _ { y } } & { \partial _ { x } } \\ { - \partial _ { z } } & { 0 } \\ { 0 } & { \partial _ { x } ^ { 2 } \partial _ { y } z \nu _ { 0 } + \partial _ { y } ^ { 3 } z \nu _ { 0 } + \partial _ { y } \partial _ { z } ^ { 2 } \nu _ { 0 } } \\ { 0 } & { - \partial _ { x } ^ { 3 } z \nu _ { 0 } - \partial _ { x } \partial _ { y } ^ { 2 } z \nu _ { 0 } - \partial _ { x } \partial _ { z } ^ { 2 } \nu _ { 0 } } \end{array} \right) . } \end{array}\tag{34}
$$

It is interesting to note that the first column of $Q _ { G T _ { 1 } } ^ { \nu _ { 0 } } ( \mathbf { x } )$ is $- { \boldsymbol { \nabla } } { \vec { A } } ,$ reflecting the gauge invariance of the vector potential. In this experiment, conditioning on $J _ { x }$ and on the boundaries of $\vec { A }$ is sufficient for solving the forward problem, but the vector potential $\vec { A }$ is a latent variable that we cannot observe. What we are actually interested in is the magnetic flux density $\vec { B } ( x , y , z )$ , that we can calculate by taking the curl of the vector potential $\vec { B } = \bar { \nabla } \times \vec { A } .$ Similar to the first experiment, we can add tasks to infer $\vec { B }$ by taking the curl of the first three rows of $Q _ { G T _ { 1 } } ^ { \phi } ( \mathbf { x } )$ , resulting in a total parametrization matrix

$$
\begin{array} { r } { { \cal Q } _ { G T _ { + B } } ^ { \nu _ { 0 } } ( { \bf x } ) = \left( \begin{array} { l l l } { - \partial _ { x } } & { \ - \partial _ { y } } \\ { - \partial _ { y } } & { \ \partial _ { x } } \\ { - \partial _ { z } } & { \ 0 } \\ { 0 } & { \ \partial _ { x } ^ { 2 } \partial _ { y } z \nu _ { 0 } + \partial _ { y } ^ { 3 } z \nu _ { 0 } + \partial _ { y } \partial _ { z } ^ { 2 } \nu _ { 0 } } \\ { 0 } & { \ - \partial _ { x } ^ { 3 } z \nu _ { 0 } - \partial _ { x } \partial _ { y } ^ { 2 } z \nu _ { 0 } - \partial _ { x } \partial _ { z } ^ { 2 } \nu _ { 0 } } \\ { 0 } & { \ \qquad - \partial _ { x } \partial _ { z } } \\ { 0 } & { \ \qquad - \partial _ { y } \partial _ { z } } \\ { 0 } & { \ \partial _ { x } ^ { 2 } + \partial _ { y } ^ { 3 } } \end{array} \right) . } \end{array}\tag{35}
$$

The PIGP method uses $J _ { y }$ and $J _ { z }$ as additional input tasks:

$$
\begin{array} { r } { \left( \begin{array} { c c c c c c } { - \partial _ { y } ^ { 2 } \nu _ { 0 } z - \partial _ { z } ^ { 2 } \nu _ { 0 } } & { \partial _ { x } \nu _ { 0 } \partial _ { y } z } & { \partial _ { z } \nu _ { 0 } \partial _ { x } } & { - 1 } & { 0 } & { 0 } \\ { \partial _ { x } \nu _ { 0 } \partial _ { y } z } & { - \partial _ { x } ^ { 2 } \nu _ { 0 } z - \partial _ { z } ^ { 2 } \nu _ { 0 } } & { \partial _ { z } \nu _ { 0 } \partial _ { y } } & { 0 } & { - 1 } & { 0 } \\ { \partial _ { z } \nu _ { 0 } \partial _ { x } } & { \partial _ { z } \nu _ { 0 } \partial _ { y } } & { - \partial _ { x } ^ { 2 } \nu _ { 0 } - \partial _ { y } ^ { 2 } \nu _ { 0 } } & { 0 } & { 0 } & { - 1 } \end{array} \right) \left( \begin{array} { c } { A _ { x } } \\ { A _ { y } } \\ { A _ { z } } \\ { J _ { x } } \\ { J _ { y } } \\ { J _ { z } } \end{array} \right) = \left( \begin{array} { c } { 0 } \\ { 0 } \\ { 0 } \end{array} \right) } \end{array}\tag{36}
$$

and the corresponding parametrization matrix (already including the curl operator to calculate $\vec { B } )$ is

$$
\mathcal Q _ { P I G P } ^ { v _ { 0 } } ( \mathbf x ) = \left( \begin{array} { c c c c } { 1 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 1 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { 1 } & { 1 } \\ { - ( \partial _ { y } ^ { 2 } z + \partial _ { z } ^ { 2 } ) v _ { 0 } } & { \partial _ { x } \partial _ { y } z v _ { 0 } } & { v _ { 0 } \partial _ { z } \partial _ { x } } \\ { \partial _ { x } \partial _ { y } z v _ { 0 } } & { - ( \partial _ { x } ^ { 2 } z + \partial _ { z } ^ { 2 } ) v _ { 0 } } & { \partial _ { z } v _ { 0 } \partial _ { y } } \\ { \partial _ { z } v _ { 0 } \partial _ { x } } & { \partial _ { z } v _ { 0 } \partial _ { y } } & { - ( \partial _ { x } ^ { 2 } + \partial _ { y } ^ { 2 } ) v _ { 0 } } \\ { 0 } & { - \partial _ { z } } & { \partial _ { y } } \\ { \partial _ { z } } & { 0 } & { - \partial _ { x } } & { - \partial _ { x } } \\ { - \partial _ { y } } & { \partial _ { x } } & { \partial _ { x } } & { 0 } \end{array} \right) .\tag{37}
$$

We choose the true parameter value in the simulations to be $\nu _ { 0 } = 1$ . Given a fixed current density ${ \vec { J } } ,$ the corresponding solution for the vector potential $\vec { A }$ and the magnetic flux density $\vec { B }$ is

$$
J = 2 \nu _ { 0 } \left( \begin{array} { c } { { z ^ { 3 } - 3 z ^ { 2 } + 2 z - y + y ^ { 2 } } } \\ { { 0 } } \\ { { 0 } } \\ { { 0 } } \end{array} \right) \overset { \vartriangle } { \vec { A } } = \left( \begin{array} { c c } { { y ( 1 - y ) ( z - 1 ) ( z - 2 ) } } \\ { { 0 } } \\ { { 0 } } \\ { { 0 } } \end{array} \right) \quad \vec { B } = \left( \begin{array} { c } { { 0 } } \\ { { ( y - y ^ { 2 } ) ( 2 z - 3 ) } } \\ { { - ( 1 - 2 y ) ( z ^ { 2 } - 3 z + 2 ) } } \end{array} \right) .\tag{38}
$$

For the forward problem, we want to infer $\vec { B }$ provided $\vec { J }$ and the boundary constraint ${ \vec { n } } \times A =$ 0 on ∂Ω. In the GT method, we encode our knowledge by conditioning on a $6 \times 6 \times 6$ grid on $J _ { x }$ and the ghost task, while in the PIGP method, we condition the tasks $J _ { x } , J _ { y }$ , and $J _ { z }$ on these

gridpoints. Per construction, $J _ { y }$ and $J _ { z }$ are enforced to be exactly zero in the GT method. For both methods, we enforced the boundary condition on the necessary surface points of the point $\mathrm { g r i d } ,$ amounting to

$$
\begin{array} { r l } & { A _ { x } ( x _ { i } , y = 0 , z _ { j } ) = A _ { x } ( x _ { i } , y = 1 , z _ { j } ) = A _ { x } ( x _ { i } , y _ { j } , z = 1 ) = A _ { x } ( x _ { i } , y _ { j } , z = 2 ) = 0 } \\ & { A _ { y } ( x = 0 , y _ { i } , z _ { j } ) = A _ { y } ( x = 1 , y _ { i } , z _ { j } ) = A _ { y } ( x _ { i } , y _ { j } , z = 1 ) = A _ { y } ( x _ { i } , y _ { j } , z = 2 ) = 0 } \\ & { A _ { z } ( x = 0 , y _ { i } , z _ { j } ) = A _ { z } ( x = 1 , y _ { i } , z _ { j } ) = A _ { z } ( x _ { i } , y = 0 , z _ { j } ) = A _ { z } ( x _ { i } , y = 1 , z _ { j } ) = 0 \quad \forall i , j = 1 , . . . , n . } \end{array}\tag{39}
$$

This yields a total of 360 boundary points for both methods and $2 \cdot 6 ^ { 3 } = 4 3 2$ inner points $( J _ { x }$ and $\xi _ { G T } )$ for GT resp. $3 \cdot 6 ^ { 3 } = 6 4 8 ( J _ { x } , J _ { y }$ , and $J _ { z } )$ for PIGP. The test set consists of a $1 8 \times 1 8 \times 1 8$ grid across all tasks.

The inverse problem is structured analogously to the previous experiments on $n ^ { 3 }$ training points on every task on a grid with $n = 3 , 5 , 7$ . As in the second experiment, we show the effect of adding extra collocation points to the ghost/input tasks in the ablation studies in Appendix B. Since the vector potential $\vec { A }$ is not a measurable quantity, we did not use it for the inverse problem.

Since this is a three-dimensional problem, Fig. 9 shows a quiver plot of the inferred magnetic flux density $\vec { B }$ of the GT method as a visualization. Fig. 10 resp. Fig. 11 show a 2d slice of the conditioned tasks resp. the inferred $\vec { B }$ fields through the $y - z$ plane at $x = 0 . 6$ . In the corner of each subplot we show the RMSE calculated over the whole volume.

As in the previous experiments we see good performances across both methods with RMSEs between 0.002 and 0.01 in the inferred magnetic flux density components. In the inverse problem setting shown in Fig. 12 however we observe a significantly better performance of GT than PIGP, which can be understood when looking at the posterior distribution of $\nu _ { 0 } \mathbf { : }$ Similar to experiment 1 (Fig. 6), PIGP has again a rather biased posterior (shown in the lowest panel of Fig. 13). This can be attributed to the lack of informativeness of the data-fit term of the log marginal likelihood (Eq. 2) shown in the middle panel of Fig. 13. The complexity penalty is similarly peaked for both methods, favouring small values of $\nu _ { 0 } .$ GT's data-fit term is able to compensate this bias by peaking sharply around the correct value $\nu _ { 0 } \approx 1$ . On the contrary, PIGPs data-fit term is flatter, i.e. less informative. The reason for this may lie in the column rank and therefore the coupling strength of the parametrization matrices, discussed in more detail in chapter 5.

![](images/d1edad95b523a03a51f1ad7163b2680e1cfc0365c0cad607963b8f2a96730fa8.jpg)  
Figure 9: Quiver plot of the magnetic flux density $\vec { B }$ inferred with GT in the forward problem of experiment 3 given Dirichlet boundary conditions for A and 6× 6×6 training points for $J _ { x }$ and the ghost task $\xi _ { G T } .$ Direction of B is indicated with arrows, the strength is represented with the length of the arrows as well as their color.

![](images/47c713e2d290f510d32ccd2c9072df911367b633f51dad6f9f44bd0f934d420e.jpg)  
Figure 10: Visualization of the conditioned tasks in the forward problem of Experiment 3. The left column shows two dimensional slices though the y – z plane at $x = 0 . 6$ of the analytic ground truth, the middle column shows the ghost tasking method, and the right column shows PIGP. The rows depict the (x,y,z) components of the vector potential A, the x component of the current density $J _ { x } ,$ and the ghost task $\xi _ { G T }$ for GT, and $J _ { y } , J _ { z }$ respectively for PIGP. Training data is marked with blue points and the RMSE over the full volume is written in the corner of each task.

![](images/e160e2b178b17df72b093cccb7d9f0fcc82b29ce5b9578ffdb63c4f56d2c1d85.jpg)  
Figure 11: Visualization of the solution of the forward problem of Experiment 3 for the magnetic flux density components. The left column shows two dimensional slices though the y – z plane at x = 0.6 of the analytic ground truth, the middle column shows the ghost tasking method, and the right column shows PIGP. The rows depict the (x,y,z) components of the inferred quantity of interest, the magnetic flux density B. No training data on B is used. The RMSE over the full volume is written in the corner of each task.

![](images/5e5126c93572fc73b00ee9366f0b7e930bedc4bb5fd98da1f2bf9c0db983544f.jpg)  
Figure 12: Relative error of the MAP estimate of the parameter v0 as performance metric of the inverse problem of experiment 3, analogously to Fig. 5 and 8. We see a clear advantage of the GT method (green) as compared to PIGP (blue) in all experiments.

![](images/6b0f6245a110a4b4442dc17e4f4d20fe98fc7389c79f389601a356c666f4a8af.jpg)

![](images/7ee8ede362f2d6a56009b502f5cf24c7d3b329ff15958c9584562d2cebac4a3f.jpg)

![](images/2933e9250b009a2d5eb1f14db3108cfe70b0849c23bb2bf4cf2c8b756e3f2d69.jpg)  
Figure 13: Re-scaled components $p - \log ( \operatorname* { m a x } ( p ) - p + 1 )$ of the log marginal likelihood $( \mathrm { E q . ~ } 2 )$ for GT (marked with x) and PIGP (marked with circles) as a function of $\nu _ { 0 } ,$ conditioned on the respective $\mathbf { M A P }$ estimates of the calibration parameters A and l for the $n = 3$ training data set and all values σ evaluated on 50 points $\nu _ { 0 } \in [ - 1 , 3 ]$ . The first row shows the complexity term, the second row shows the data-fit term, and the last row shows the total log marginal likelihood Curves are not normalized due to re-scaling; the relevant information is conveyed by the position and width of the peaks.

## 5. Discussion and Qualitative Comparison

## 5.1. Discussion of experimental results

A central structural difference between ghost tasking and PIGP lies in the number of latent functions required to parametrize the solution space, and this difference is a direct consequence of the task-counting results established in chapter 3.2. Recall that the PIGP construction of Raissi et al. [9] always requires $q$ additional inhomogeneity tasks, and the resulting minimal parametrization matrix $\boldsymbol { Q } _ { P I G P } ^ { \phi }$ correspondingly has $q$ columns, i.e., q latent functions. Ghost tasking, by contrast, requires at most m $( m \in \{ 1 , 2 \} )$ tasks additional to the $q \leq p \leq 2 q - m$ tasks from the system matrix $R ,$ so that the number of latent functions $n = p + m - q$ ranges between m and $q ,$ with the upper bound recovering PIGP exactly (chapter 3.3). Fewer latent functions mean that a fixed amount of training data constrains each latent function more strongly, resulting in a more sharply peaked, informative data-fit term.

We observe this effect across our experiments. In Experiment 3, the most extreme case, the parametrization matrices $Q _ { G T _ { + B } } ^ { \nu _ { 0 } }$ and $\boldsymbol { Q } _ { P I G P } ^ { \nu _ { 0 } }$ (Eq. (35) and (37)) illustrate this directly. Since the first three tasks, corresponding to ${ \vec { A , } }$ are not observed in the inverse problem, $Q _ { G T _ { + B } } ^ { \nu _ { 0 } }$ effectively has column rank $n = 1$ : a single latent function is sufficient to fit all training data across five tasks. This constrains the latent function strongly, yielding a data-fit term sharply peaked around the true value of $\nu _ { 0 }$ (Fig. 13). In contrast, $\mathcal { Q } _ { P I G P } ^ { \nu _ { 0 } }$ has column rank 3, so three latent functions must be fit from training data across six tasks. Despite the larger total amount of training data, each latent function is effectively less constrained, allowing misfits in $\nu _ { 0 }$ to be compensated for and producing a flatter, less informative data-fit term that fails to fully counteract the complexity penalty.

The same effect, though weaker, appears in Experiments 1 and 2: in Experiment 1, $Q _ { G T } ^ { \alpha }$ (Eq. (19)) has column rank 2, while $Q _ { P I G P } ^ { \alpha }$ (Eq. (21)) has column rank 3, and the broader posteriors of PIGP in Fig. 6 reflect this effect. In Experiment 2, $Q _ { G T } ^ { ( \ell , g ) }$ (Eq. (27)) has column rank 1, while $Q _ { P I G P } ^ { ( \ell , g ) } ( \mathrm { E q . } ( 2 8 ) )$ has column rank 2.

## 5.2. Practical considerations and limitations

Beyond the structural aspects discussed above, some practical considerations arise when applying ghost tasking.

First, finding a suitable generator Λ is algorithmic for holonomic systems (meaning the dimension of the system equals the number of variables), where it can be computed directly using Macaulay2 [27] (Appendix $\mathbf { A } )$ . More generally, Stafford's two-generator theorem (Th. 2.1) admits a constructive proof [52, 53, 54] and while they are in principle implemented in OreModules's Stafford package and Macaul $\mathsf { a y 2 5 }$ Dmodules package, to our knowledge, they can currently not be applied to extension modules in those algebra programs. In practice, we therefore rely on a heuristic ansatz for Λ(x), that we verify computationally (see chapter 3.3 and Appendix A).

Second, ghost tasking has the advantage that the ghost tasks are known to be exactly zero, so collocation points can be added freely without requiring measurements or real data. However, adding collocation points trades off accuracy against computation time, which scales cubically with the total amount of training data. For oscillatory systems, we observed that additional collocation points yield substantial improvements in accuracy (Fig. 5); for the remaining experiments, the improvement is comparatively modest (Tab. B.2 and B.3 in Appendix B).

Third, as with any parametrization method applied to systems with non-constant coefficients, the resulting parametrized kernels can become numerically ill-conditioned, which may affect stability in sharply peaked likelihood regimes [55]. We refer to the extensive literature on stabilization and regularization techniques, e.g. [56], for approaches to mitigating this issue.

Lastly, a practical trade-off concerns the choice between the projective and the torsion-free variant of ghost tasking introduced in chapter 3.3. Although the two variants are largely comparable in predictive performance, requiring projectivity incurs greater computational overhead both in the computer algebra required to determine Λ(x) and the parametrization matrix, and in evaluation time, owing to the heuristically more complex terms arising in the projective parametrization. We therefore recommend using the weaker, torsion-free variant in practice. This choice carries the theoretical caveat that parametrizability of the full solution space is no longer guaranteed, but at least in our experiments, we did not encounter a case in which this limitation was practically relevant.

## 5.3. Qualitative Method Comparison

To contextualize ghost tasking within existing hybrid GP methods, we need to compare the methods with respect to several metrics: (i) reliance on analytical prior knowledge versus algorithmic construction, (ii) structural restrictions on the admissible systems, notably parametrizability and classes of coefficients, and (iii) the trade-off between model complexity and performance, particularly in inverse problems.

Mercer-type kernels, such as those discussed in [15, 2, 3, 13], can achieve strong performance both in inverse and in forward problems, when applicable. However, they rely on prior analytical knowledge of the system in that the kernel is constructed from the form of the fundamental solutions. This is also the case for latent-force models [14], which use the system's Green's function These approaches are therefore non-algorithmic and tailored to specific problem classes where advanced prior knowledge beyond the general form of the equation is available or obtainable. While highly effective in settings where such knowledge is available, their applicability is inherently limited by the availability of such fundamental solutions; moreover, latent force models, as one of the earliest approaches, have since been outperformed by more recent methods such as LODE-GPs ("Linear Ordinary Differential Equations") [11] that construct kernels based on the Smith normal form of the system matrix and are applicable to ordinary systems with constant coefficients.

Parametrization-based methods such as PIGPs [9] and the method of [10], as well as our ghost tasking, enforce constraints through operator structure on multiple correlated tasks (outputs, and inhomogeneities or ghost tasks) and find solutions through subsequent pointwise conditioning, rather than requiring known functions. Their key advantage is that they are algorithmic, i.e., applicable without requiring analytical knowledge of the solution. A central trade-off in these methods lies in the number of auxiliary tasks and with it latent functions as discussed previously. Increasing the number of tasks improves generality by introducing more latent functions, but weakens the coupling between the tasks due to the larger latent function space. This can degrade performance in inverse problems (as seen for example in [25]), and requires more training points to constrain the additional tasks and latent functions. Conversely, reducing the number of tasks and latent functions strengthens coupling and improves data efficiency, but risks restricting admissible systems and applicability of the method to parametrizable systems. Ghost tasking can be understood as the sweet spot in this trade-off: it preserves strong coupling while relaxing parametrizability constraints for general systems.

Ghost tasking is theoretically applicable to a broad class of systems, including those with meromorphic coefficients. In practice, however, this generality is, as in [10], mediated by the need to compute parametrizations via computer algebra programs. As a consequence, it inherits the limitations of currently available computer algebra implementations, which are largely restricted to polynomial or rational coefficients (e.g., via Weyl algebras). Other approaches that seek to circumvent or eliminate the parametrizability constraints are "linear ordinary differential equation Gaussian processes" LODE-GPs [11] for ODEs and "Ehrenpreis-Palamodov Gaussian Processes" EPGPs [12] for PDEs, but they are only applicable to systems with constant coefficients.

In certain cases, the limitation to rational coefficients due to the algorithms available in computer algebra programs that we use in ghost tasking can be mitigated in practice in so-called commuting systems, where coefficient functions commute with differential operators. This allows the use of simpler algorithms over commutative polynomial rings as used in [10], yielding tractable parametrizations when such assumptions are justified. However, this approach introduces implicit structural constraints through the need to verify $R ^ { \phi } ( { \bf x } ) Q ^ { \phi } ( { \bf x } ) \stackrel { ! } { = } 0$ considering the specific coefficient functions explicitly and is therefore only applicable in restricted settings (see Appendix C for an example).

In future work it may be interesting to investigate parametrizations based on the Jacobson Normal Form (JNF) [57], analogous to the Smith Normal Form employed in [11]. While JNF applies only to ODEs, it may be able to work with general coefficient classes and reduce the latent function space of ODE systems even further. Another promising direction is the integration of ghost tasking with approaches that incorporate boundary conditions directly into the kernel. Since we have shown that ghost tasking shares the structure of the parametrized GPs of [10] and [9], boundary-condition constructions that have been applied to either of these methods, such as [19] or [20], could likely be transferred with only minor modifications. More broadly, extending parametrization-based methods to the wider class of weak formulations of differential operators remains an important direction for future research.

## 6. Conclusion

We have introduced ghost tasking as a method to extend parametrizability to physical systems that on their own are not parametrizable, while retaining a minimal latent function space. We do so by adding one, but at most two, additional ("ghost") tasks. This results in a minimal latent function space that is able to fit all solutions of the system. We show that the physics-informed Gaussian Process (PIGP) method [9] can itself be understood as a form of parametrization, alas one that introduces an often unnecessarily large number of auxiliary tasks and latent functions, rather than the minimal number for both that ghost tasking identifies. As such, ghost tasking can be understood as the optimally data-efficient parametrization on the spectrum between PIGP as in [9] and the null space-based parametrization as in [10]. We showed that ghost tasking achieves performance comparable to PIGP on forward problems while requiring fewer training data and outperforms PIGP when using the same amount of training data. On inverse problems, ghost tasking frequently attains several orders of magnitude higher accuracy, particularly in the lowdata regime. Our results demonstrate that a substantially broader class of differential equation systems can be encoded more efficiently than previously possible as priors in GPs, requiring fewer tasks and latent functions and less training data without sacrificing predictive performance. As advances in constructive algebraic analysis and automated generator construction continue, we expect the range of systems that can be encoded optimally by ghost tasking to grow further, making the framework increasingly practical for more complex physical models.

## 7. Acknowledgements

J.M. would like to thank A. Quadrat for helpful discussions and A. Leykin for providing the foundation of the Macaulay2 code snippets. J.M. and C.A. were financially supported by the Austrian Research Promotion Agency (FFG) project VENTUS within the AI for Green funding program, Grant no. 910263. S.R. was financially supported by the Austrian Science Fund (FWF), Grant DOI: 10.55776/J4774. This work has been carried out within the framework of the EUROfusion Consortium, funded by the European Union via the Euratom Research and Training Programme (Grant Agreement No 101052200 — EUROfusion). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Commission. Neither the European Union nor the European Commission can be held responsible for them.

## 8. Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used Claude (Anthropic) and GPT (OpenAI) in order to help with code development, brainstorming, and text polishing. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## References

[1] L. P. Swiler, M. Gulian, A. L. Frankel, C. Safta, J. D. Jakeman, A survey of constrained gaussian process regression: Approaches and implementation challenges, Journal of Machine Learning for Modeling and Computing 1 (2) (2020).

[2] A. Dong, Kriging Variables that Satisfy the Partial Differential Equation δZ = Y, in: M. Armstrong (Ed.), Geostatistics, Springer Netherlands, Dordrecht, 1989, pp. 237–248. doi:10.1007/978-94-015-6844-9\_17.

[3] A. Dong, Estimation géostatistique des phénomènes régis par des équations aux dérivées partielles, Ph.D. thesis, ENSMP (1990).

[4] F. J. Narcowich, J. D. Ward, Generalized hermite interpolation via matrix-valued conditionally positive definite functions, Mathematics of Computation 63 (208) (1994) 661–688.

[5] N. Wahlström, M. Kok, T. B. Schön, F. Gustafsson, Modeling magnetic fields using gaussian processes, in: 2013 IEEE International Conference on Acoustics, Speech and Signal Processing, 2013, pp. 3522–3526. doi:10.1109/ICASSP.2013.6638313.

[6] N. Wahlström, Modeling of magnetic fields and extended objects for localization applications, Linkopings Universitet, 2015.

[7] R. Berlinghieri, B. L. Trippe, D. R. Burt, R. Giordano, K. Srinivasan, T. Özgökmen, J. Xia, T. Broderick, Gaussian processes at the helm (holtz): A more fluid model for ocean currents, arXiv preprint arXiv:2302.10364 (2023).

[8] C. Jidling, N. Wahlström, A. Wills, T. B. Schön, Linearly constrained gaussian processes, Advances in neural information processing systems 30 (2017).

[9] M. Raissi, P. Perdikaris, G. E. Karniadakis, Machine learning of linear differential equations using Gaussian processes, Journal of Computational Physics 348 (2017) 683–693. doi:10.1016/j.jcp.2017.07.050. URL https://www.sciencedirect.com/science/article/pii/ S0021999117305582

[10] M. Lange-Hegermann, Algorithmic Linearly Constrained Gaussian Processes, in: Advances in Neural Information Processing Systems, Vol. 31, Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper/2018/hash/ 68b1fbe7f16e4ae3024973f12f3cb313-Abstract.html

[11] A. Besginow, M. Lange-Hegermann, Constraining Gaussian processes to systems of linear ordinary differential equations, Advances in Neural Information Processing Systems 35 (2022) 29386–29399. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ bcef27c5825d1ed8757290f237b2d851-Abstract-Conference.html

[12] M. Harkonen, M. Lange-Hegermann, B. Raita, Gaussian process priors for systems of linear partial differential equations with constant coefficients, in: International conference on machine learning, PMLR, 2023, pp. 12587–12615. URL https://proceedings.mlr.press/v202/harkonen23a.html

[13] K. G. van den Boogaart, Kriging for processes solving partial differential equations, Proceedings of the IAMG2001, Cancun, Mexiko (2001) 10–12

[14] M. A. Álvarez, D. Luengo, N. D. Lawrence, Linear Latent Force Models Using Gaussian Processes, IEEE Transactions on Pattern Analysis and Machine Intelligence 35 (11) (2013) 2693-2705. doi:10.1109/TPAMI.2013.86. URL https://ieeexplore.ieee.org/abstract/document/6514873

[15] C. G. Albert, Gaussian Processes for Data Fulfilling Linear Differential Equations, Proceedings 33 (1) (2019) 5, number: 1 Publisher: Multidisciplinary Digital Publishing Institute. doi:10.3390/proceedings2019033005. URLhttps://www.mdpi.com/2504-3900/33/1/5

[16] L. Ding, S. Mak, C. Wu, Bdrygp: a new gaussian process model for incorporating boundary information, arXiv preprint arXiv:1908.08868 (2019).

[17] M. H. Y. Tan, Gaussian process modeling with boundary information, Statistica Sinica (2018) 621–648.

[18] A. Solin, M. Kok, Know your boundaries: Constraining gaussian processes by variational harmonic features, in: K. Chaudhuri, M. Sugiyama (Eds.), Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, Vol. 89 of Proceedings of Machine Learning Research, PMLR, 2019, pp. 2193–2202. URL https://proceedings.mlr.press/v89/solin19a.html

[19] M. Lange-Hegermann, Linearly constrained gaussian processes with boundary conditions, in: A. Banerjee, K. Fukumizu (Eds.), Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, Vol. 130 of Proceedings of Machine Learning Research, PMLR, 2021, pp. 1090–1098. URL https://proceedings.mlr.press/v130/lange-hegermann21a.html

[20] M. Gulian, A. Frankel, L. Swiler, Gaussian process regression constrained by boundary value problems, Computer Methods in Applied Mechanics and Engineering 388 (2022) 114117.

[21] Y. Chen, B. Hosseini, H. Owhadi, A. M. Stuart, Solving and learning nonlinear pdes with gaussian processes, Journal of Computational Physics 447 (2021) 110668. doi:https://doi.org/10.1016/j.jcp.2021.110668. URL https://www.sciencedirect.com/science/article/pii/ S0021999121005635

[22] M. Raissi, P. Perdikaris, G. E. Karniadakis, Numerical gaussian processes for timedependent and nonlinear partial differential equations, SIAM Journal on Scientific Computing 40 (1) (2018) A172–A198. arXiv:https://doi.org/10.1137/17M1120762, doi:10.1137/17M1120762. URL https://doi.org/10.1137/17M1120762

[23] O. Hamelijnck, A. Solin, T. Damoulas, Physics-informed variational state-space gaussian processes, Advances in Neural Information Processing Systems 37 (2024) 98505–98536.

[24] H. Moss, L. Astfalck, T. Cowperthwaite, C. Doumont, S. Willis, P. Hennig, C. Nemeth, A. Zammit-Mangion, Conditioning gaussian processes on almost anything, arXiv preprint arXiv:2605.21041 (2026).

[25] J. Moser, C. Albert, S. Ranftl, Parameter learning with physics-consistent gaussian processes, in: 44th International Workshop on Bayesian Inference and Maximum Entropy Methods in Science and Engineering, Auckland, New Zealand, 2026. doi:10.5281/ zenodo.20392038. URL https://doi.org/10.5281/zenodo.20392038

[26] F. Chyzak, A. Quadrat, D. Robertz, Oremodules: A symbolic package for the study of multidimensional linear systems, in: Applications of time delay systems, Springer, 2007, pp. 233–264.

[27] D. R. Grayson, M. E. Stillman, Macaulay2, a software system for research in algebraic geometry, Available at http://www2.macaulay2.com.

[28] Rasmussen, Gaussian Processes for Machine Learning Vol. 2, MIT press Cambridge, MA, 2006.

URL https://mitpress.mit.edu/9780262182539/ gaussian-processes-for-machine-learning/

[29] A. Quadrat, D. Robertz, A constructive study of the module structure of rings of partial differential operators, Acta Applicandae Mathematicae 133 (1) (2014) 187–234.

[30] M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics 378 (C) (11 2018). doi : 10 .1016/ j.jcp.2018.10.045. URL https://www.osti.gov/biblio/1595805

[31] A. Quadrat, An introduction to constructive algebraic analysis and its applications, Les cours du CIRM 1 (2) (2010) 281–471.

[32] C. Schnell, D-modules, lecture notes (2019). URLhttps://www.math.stonybrook.edu/\~cschnell/pdf/notes/d-modules. pdf

[33] J. J. Rotman, An introduction to homological algebra, no. 2, Springer, 2009.

[34] J. T. Stafford, Module structure of Weyl algebras, Journal of the London Mathematical Society. Second Series 18 (3) (1978) 429–442. doi:10.1112/j1ms/s2-18.3.429.

[35] T. Cluzeau, A. Quadrat, Serre's reduction of linear partial differential systems with holonomic adjoints, Journal of symbolic computation 47 (10) (2012) 1192–1213.

[36] J.-F. Pommaret, A. Quadrat, A functorial approach to the behaviour of multidimensional control systems, International Journal of Applied Mathematics and Computer Science 13 (1) (2003) 7–13.

[37] U. Oberst, Multidimensional constant linear systems, Acta Applicandae Mathematica 20 (1) (1990) 1–175.

[38] A. Quadrat, Grade filtration of linear functional systems, Acta Applicandae Mathematicae 127 (1) (2013) 27–86.

[39] J. Willems, S. Mitter, Controllability, observability, pole allocation, and state reconstruction, IEEE Transactions on Automatic Control 16 (6) (1971) 582–595. doi:10.1109/ TAC.1971.1099819.

[40] M. Fliess, Some basic structural properties of generalized linear systems, Systems & Control Letters 15 (5) (1990) 391–396. doi:https://doi.org/10.1016/0167-6911(90) 90062-Y. URL https://www.sciencedirect.com/science/article/pii/ 016769119090062Y

[41] E. Zerz, Topics in multidimensional linear systems theory, Springer, 2000.

[42] J. Pommaret, A. Quadrat, Algebraic analysis of linear multidimensional control systems, IMA Journal of Mathematical control and Information 16 (3) (1999) 275–297.

[43] F. Chyzak, A. Quadrat, D. Robertz, Effective algorithms for parametrizing linear control systems over ore algebras, Applicable Algebra in Engineering, Communication and Computing 16 (5) (2005) 319–376.

[44] M. S. Boudellioua, A. Quadrat, Serre's reduction of linear functional systems, Mathematics in Computer Science 4 (2) (2010) 289–312.

[45] E. Noether, Abstrakter aufbau der idealtheorie in algebraischen zahl- und funktionenkörpern, Mathematische Annalen 96 (1927) 26–61. URL http://eudml.org/doc/159155

[46] L. M. de Moura, S. Ullrich, The lean 4 theorem prover and programming language, in: CADE, 2021. URL https://api.semanticscholar.org/CorpusID:235800962

[47] Christopher Albert, Stafford 3.8, Palomar, PALOMAR-2026-09-05-000007 v2 (2026). URL https://palomar-registry.org/entry?id=PALOMAR-2026-09-05-000007& version=2

[48] J. R. Gardner, G. Pleiss, D. Bindel, K. Q. Weinberger, A. G. Wilson, Gpytorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration, in: Advances in Neural Information Processing Systems, 2018.

[49] J. R. Gardner, G. Pleiss, D. Bindel, K. Q. Weinberger, A. G. Wilson, GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration, arXiv:1809.11165 [cs] (Jun.2021). doi:10.48550/arXiv.1809.11165. URL http://arxiv.org/abs/1809.11165

[50] J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, Q. Zhang, JAX: composable transformations of Python+NumPy programs, http://github. com/jax-ml/jax, version 0.3.13 (2018).

[51] C. G. Albert, P. Lainer, O. Biro, 2d fourier finite element formulation for magnetostatics in curvilinear coordinates with a symmetry direction, Computer Physics Communications 277 (2022) 108401.

[52] A. Leykin, Algorithmic proofs of two theorems of stafford, Journal of Symbolic Computation 38 (6) (2004) 1535–1550.

[53] A. Hillebrand, W. Schmale, Towards an effective version of a theorem of stafford, Journal of Symbolic Computation 32 (6) (2001) 699–716.

[54] A. Quadrat, D. Robertz, Constructive computation of flat outputs of a class of multidimensional linear systems with variable coefficients, in: Proceedings of MTNS, 2006, pp. 24–28.

[55] J. Moser, C. Albert, S. Ranftl, Resolving positive semi-definiteness in physics-informed kernels for scientific machine learning (2026). arXiv:2608.20985. URL https://arxiv.org/abs/2608.20985

[56] M. Benning, M. Burger, Modern regularization methods for inverse problems, Acta numerica 27 (2018) 1–111.

[57] N. Jacobson, Basic algebra I, Courier Corporation, 2012.

## Appendix A. Computer algebra syntax

Here we provide the Macaulay2 and OreModules syntax used to construct a ghost column Λ(x) and to check the validity of a proposed one. For the full scripts, we refer to our github page. As before, we drop the index φ for physical parameters and the argument (x) for readability. Full tutorials and detailed discussions on OreModules can be found on their website³.

Deining the algebra and the system.   
--Macaulay   
needsPackage "Dmodules"   
D = frac(QQ[a])[t, x, dt, dx, WeylAlgebra => {x=>dx, t=> dt}]   
R = matrix{{x\*dx + dt, a\*dx}, {0, x\*dx + dt}}   
--OreModules   
with(OreModules):   
with(Ore\_algebra):   
D := DefineOreAlgebra(diff = [dt, t], diff = [dx, x],   
polynom = {t, x}, comm = {a})   
R := evalm([[dt + x\*dx, a\*dx], [0, dt + x\*dx]])

Here hides a common pitfall: a system may be parametrizable for generic values of commuting constants (such as a here), but the parametrization fails for a special combination of them (such as in [11]). Both packages here certify only the generic case, so if the constants are known numerically beforehand, it is safer to substitute them before certifying, precisely because a special value can break a generically parametrizable system.

Finding Λ. The columns of Λ must be chosen so that their residues generate this module (cf. Theorem 3.2). If ext1(M, D) is holonomic, it is automatically cyclic [35], and Macaulay2 can produce a cyclic generator algebraically:

ext1M = coker Dtransposition R;   
isHolonomic ext1M --if true, the next line works   
Lambda = Dtransposition (makeCyclic Dtransposition R).Generator

Dtransposition implements the involution map needed to map from left to right modules and inversely, needed to use Macaulay2's native Gröbner machinery.

For a non-holonomic ext1(M, D) (as in experiment 1 and 3), we do not know of a constructive method that is implemented for extension modules. The fallback is to loop through a set of candidate operators and test each candidate with the projectivity or torsion-freeness check given below, stopping at the first success; this has worked reasonably well in practice. The script running the loop can be found on our github page.

Once we have a candidate, we define P = (R −Λ).

Checking torsion-freeness. M is torsion-free iff ex $\mathfrak { t } _ { D } ^ { 1 } ( N , D ) = 0 ( ( 1 )$ of Lemma 2.3). In Macaulay2, we explicitly calculate ex $\mathsf { t } _ { D } ^ { 1 } ( N , D )$

strip = X -> map((ring X)^(numRows X), (ring X)^(numColumns X), entries X)   
K1 := strip syz P;   
Q1 := strip transpose Dtransposition P;   
Q2 := strip transpose Dtransposition K1;   
ext1N := prune (kernel Q2 / image Q1);   
ext1N == 0 --if true, then M is torsion-free   
In OreModules the analogous call is   
Exti(Involution(P, D), D, 1);   
if the first of its three returned matrices is the identity, M is torsion-free.

Checking projectivity. To test projectivity, it is actually easiest to test if M is stably free, and thus by implication also projective. Lemma 2.3 tells us, that M is stably free iff ex $\mathrm { t } _ { D } ^ { 1 } ( M , D ) = 0 .$ In Macaulay2 we can test this by checking whether $D ^ { q } \subseteq P D ^ { p + m }$ , i.e. whether $\mathrm { e x t } _ { D } ^ { 1 } ( M , D ) =$ $D ^ { q } / P D ^ { p + m } = 0$

isSubset(image id\_(target Dtransposition P), image Dtransposition P)   
Alternatively, we can test if R has a right inverse S with Alternatively, we can test if R has a right inverse S with $P S = \operatorname { i d } _ { q } \left( ( 3 \right)$ of Lemma 2.3). of Lemma 2.3).

W = id\_(target Dtransposition P) // Dtransposition P   
W =!= null and (Dtransposition P)\*W == id\_(target Dtransposition P)   
S = Dtransposition W   
In OreModules, the analogous call is to calculate   
RightInverse(P, D)

Alternatively, we can check (2) of Lemma 2.3 by calculating the first n extension modules $\mathrm { e x t } _ { D } ^ { i } ( N , D )$ (with n being the number of variables in the chosen Weyl algebra $A _ { n } ( \mathbb { R } ) )$ 1

```javascript
Exti(Involution(R, D), D, n);
```

the first matrix returned by each of the n calls is the presentation matrix $R _ { \mathrm { e x t } ^ { i } }$ of $\mathrm { e x t } _ { D } ^ { i } ( M , D ) ;$ if every first matrix is the identity, all n extension modules vanish and the system is projective.

Minimal parametrization. Once we have checked torsion-freeness or projectivity, we need to calculate the corresponding parametrization matrix Q. In Macaulay2, we can build a parametrization matrix by calculating the generators of the right null space of P

Q = Dtransposition mingens image syz Dtransposition P   
transpose Q\*(transpose P) -- test: should be 0

We note that Macaulay2 has a reversed product convention, so to check ${ \overline { { P Q = 0 } } } .$ we actually need to check $Q ^ { T } P ^ { T } = 0 .$ Based on this parametrization, we can also check torsion-freeness analogous to what [10] has proposed for commutative algebras by calculating the left null space $P _ { n e w }$ of Q an checking if it presents the same module as P does:

Pnew = transpose mingens image syz transpose Q

We note that the parametrization that we can calculate with Macaulay2 does not necessarily have the minimal number of columns. In OreModules, we can find a minimal parametrization in OreModules by calling

MinimalParametrization(P, D)

Table B.1: Ablation study experiment 1. Median absolute error of the physical parameters (each cell: $\begin{array} { r } { \frac { \Delta \ell } { \ell } \ / \ \frac { \Delta g } { g } \ / \ \frac { \Delta g / \ell } { g / \ell } ) } \end{array}$ relative to their true values $( \ell , g ) = ( 2 . 5 , 9 . 8 1 )$ across 10 runs. We compare the variants "standard", where every task has the same amount of training data n, and $" \mathrm { + " , }$ where the ghost task is constrained on 25 points and the standard deviation of the likelihood in these tasks is fixed to be $\sigma = 0 . 0 0 1$ . We compare two settings "measured $f ( t ) " ,$ where $f ( t )$ is provided with n measurements, and "known $f ( t ) "$ , where f(t) is provided with 25 noise-free input points for all settings. For PIGP, we cannot provide $\mathrm { ~ a ~ } " + "$ variant for "measured $f ( t ) "$ per definition. Overall performance is calculated using the geometric mean of the above results and shown in the last row. We see similar performances between GT and PIGP, while $" \mathrm { G T } + "$ outperforms both drastically especially for $g / \ell$ in the measured case. For known $f ( t ) ,$ performances are similar.
<table><tr><td rowspan="3" colspan="2"> $\frac { \Delta \ell } { \ell } / \frac { \Delta g } { g } / \frac { \Delta g / \ell } { g / \ell }$ </td><td colspan="4">measured f(t)</td><td colspan="2">known f(t)</td></tr><tr><td colspan="2">GT</td><td>PIGP</td><td colspan="2">GT</td><td>PIGP</td></tr><tr><td></td><td>standard</td><td>+</td><td>standard</td><td></td><td></td></tr><tr><td>3</td><td>0.001</td><td> $6 . 0 \times 1 0 ^ { - 1 } / 2 . 3 \times 1 0 ^ { - 1 } / 2 . 3 \times 1 0 ^ { - 1 }$ </td><td> $5 . 5 \times 1 0 ^ { - 2 } / 5 . 5 \times 1 0 ^ { - 2 } / 4 . 5 \times 1 0 ^ { - 5 }$ </td><td> $6 . 6 \times 1 0 ^ { - 2 } / 8 . 6 \times 1 0 ^ { - 1 } / 8 . 5 \times 1 0 ^ { - 1 }$ </td><td> $3 . 1 \times 1 0 ^ { - 4 } / 3 . 5 \times 1 0 ^ { - 4 } / 3 . 5 \times 1 0 ^ { - 5 }$ </td><td> $3 . 3 \times 1 0 ^ { - 4 } / 3 . 7 \times 1 0 ^ { - 4 } / 3 . 3 \times 1 0 ^ { - 5 } .$ </td></tr><tr><td>3</td><td>0.01</td><td>6.0 × 10−1/2.3 × 10−1/2.3 × 10−1</td><td> $2 . 6 \times I O ^ { - i } / 2 . 6 \times 1 0 ^ { - 1 } / 7 . 7 \times 1 0 ^ { - 4 }$ </td><td>6.6 × 10−2/8.6 × 10−1/8.5 × 10−1</td><td> $5 . 6 \times 1 0 ^ { - 3 } / 5 . 1 \times 1 0 ^ { - 3 } / 4 . 5 \times 1 0 ^ { - 4 }$ </td><td> $5 . 6 \times 1 0 ^ { - 3 } / 5 . 1 \times 1 0 ^ { - 3 } / 4 . 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>3</td><td>0.1</td><td> $6 . 0 \times 1 0 ^ { - 1 } / 2 . 3 \times 1 0 ^ { - 1 } / 2 . 3 \times 1 0 ^ { - 1 }$ </td><td> $\phantom { - } 5 . 7 \times I O ^ { - 1 } / 5 . 8 \times 1 0 ^ { - 1 } / 2 . 2 \times 1 0 ^ { - 3 }$ </td><td> $6 . 8 \times 1 0 ^ { - 2 } / 8 . 6 \times 1 0 ^ { - 1 } / 8 . 5 \times 1 0 ^ { - 1 }$ </td><td> $1 . 1 \times 1 0 ^ { - 1 } / 1 . 0 \times 1 0 ^ { - 1 } / 7 . 0 \times 1 0 ^ { - 3 }$ </td><td> $9 . 6 \times 1 0 ^ { - 2 } / 8 . 9 \times 1 0 ^ { - 2 } / 4 . 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>3</td><td>0.2</td><td> $6 . 0 \times 1 0 ^ { - 1 } / 2 . 4 \times 1 0 ^ { - 1 } / 2 . 3 \times 1 0 ^ { - 1 }$ </td><td> $\boldsymbol { 5 . 9 \times 1 0 ^ { - 1 } } / 6 . 0 \times 1 0 ^ { - 1 } / 7 . 8 \times 1 0 ^ { - 3 }$ </td><td> $6 . 4 \times 1 0 ^ { - 2 } / 8 . 7 \times 1 0 ^ { - 1 } / 8 . 6 \times 1 0 ^ { - 1 }$ </td><td> $2 . 3 \times 1 0 ^ { - 1 } / 2 . 1 \times 1 0 ^ { - 1 } / 1 . 8 \times 1 0 ^ { - 2 }$ </td><td> $2 . 2 \times 1 0 ^ { - 1 } / 2 . 0 \times 1 0 ^ { - 1 } / 1 . 5 \times 1 0 ^ { - 2 }$ </td></tr><tr><td></td><td>0.001</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>5</td><td>0.01</td><td> $7 . 1 \times 1 0 ^ { - 1 } / 7 . 2 \times 1 0 ^ { - 1 } / 3 . 6 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 5 . 5 \times 1 0 ^ { - 2 } } / 5 . 5 \times 1 0 ^ { - 2 } / 5 . 9 \times 1 0 ^ { - 5 }$ </td><td> $4 . 9 \times 1 0 ^ { - 1 } / 4 . 2 \times 1 0 ^ { - 1 } / 1 . 3 \times 1 0 ^ { - 1 }$ </td><td> $5 . 1 \times 1 0 ^ { - 4 } / 5 . 0 \times 1 0 ^ { - 4 } / 5 . 7 \times 1 0 ^ { - 5 }$ </td><td> $\mathbf { 5 . 1 \times 1 0 ^ { - 4 } } / 5 . 0 \times 1 0 ^ { - 4 } / 5 . 6 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>5</td><td>0.1</td><td> $7 . 1 \times 1 0 ^ { - 1 } / 7 . 2 \times 1 0 ^ { - 1 } / 3 . 8 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 1 . 9 \times 1 0 ^ { - 2 } } / \mathbf { 1 . 8 \times 1 0 ^ { - 2 } } / 6 . 3 \times 1 0 ^ { - 4 }$ </td><td> $4 . 9 \times 1 0 ^ { - 1 } / 4 . 2 \times 1 0 ^ { - 1 } / 1 . 3 \times 1 0 ^ { - 1 }$ </td><td> $1 . 9 \times 1 0 ^ { - 3 } / 1 . 3 \times 1 0 ^ { - 3 } / 6 . 5 \times 1 0 ^ { - 4 }$ </td><td> $2 . 0 \times 1 0 ^ { - 3 } / 1 . 2 \times 1 0 ^ { - 3 } / 6 . 5 \times 1 0 ^ { - 4 }$   $6 . 5 \times 1 0 ^ { - 2 } / 5 . 8 \times 1 0 ^ { - 2 } / 7 . 6 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>5</td><td>0.2</td><td> $6 . 8 \times 1 0 ^ { - 1 } / 7 . 3 \times 1 0 ^ { - 1 } / 1 . 3 \times 1 0 ^ { - 1 }$   $6 . 9 \times 1 0 ^ { - 1 } / 7 . 4 \times 1 0 ^ { - 1 } / 1 . 6 \times 1 0 ^ { - 1 }$ </td><td> $6 . 8 \times I O ^ { - I } / 6 . 8 \times I O ^ { - I } / 5 . 3 \times 1 0 ^ { - 3 }$   $6 . 6 \times I O ^ { - I } / 6 . 6 \times I O ^ { - I } / 1 . 1 \times 1 0 ^ { - 2 }$ </td><td> $\bar { 5 } . 2 \times 1 0 ^ { - 1 } / 4 . 5 \times 1 0 ^ { - 1 } / 1 . 3 \times 1 0 ^ { - 1 }$   $^ { - 1 } / 4 . 9 \times 1 0 ^ { - 1 } / 1 . 3 \times 1 0 ^ { - 1 }$ </td><td> $7 . 3 \times 1 0 ^ { - 2 } / 6 . 7 \times 1 0 ^ { - 2 } / 7 . 3 \times 1 0 ^ { - 3 }$   $2 . 2 \times 1 0 ^ { - 1 } / 2 . 0 \times 1 0 ^ { - 1 } / 1 . 3 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 2 . 0 \times 1 0 ^ { - 1 } } / 1 . 8 \times 1 0 ^ { - 1 } / 1 . 2 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>5</td><td></td><td></td><td></td><td>5.5 × 10</td><td></td><td></td></tr><tr><td>10</td><td>0.001 0.01</td><td> $5 . 5 \times 1 0 ^ { - 4 } / 5 . 3 \times 1 0 ^ { - 4 } / 6 . 4 \times 1 0 ^ { - 5 }$ </td><td> $3 . 5 \times 1 0 ^ { - 4 } / 3 . 9 \times 1 0 ^ { - 4 } / 3 . 3 \times 1 0 ^ { - 5 }$ </td><td> $4 . 3 \times 1 0 ^ { - 4 } / 4 . 2 \times 1 0 ^ { - 4 } / 2 . 4 \times 1 0 ^ { - 5 }$ </td><td> $3 . 0 \times 1 0 ^ { - 4 } / 3 . 3 \times 1 0 ^ { - 4 } / 2 . 7 \times 1 0 ^ { - 5 }$ </td><td> $\mathbf { 3 . 0 \times 1 0 ^ { - 4 } } / 3 . 4 \times 1 0 ^ { - 4 } / 2 . 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>10</td><td>0.1</td><td> $2 . 5 \times 1 0 ^ { - 3 } / 2 . 5 \times 1 0 ^ { - 3 } / 3 . 9 \times 1 0 ^ { - 4 }$ </td><td> $\mathbf { 2 . 0 \times 1 0 ^ { - 3 } } / 2 . 0 \times 1 0 ^ { - 3 } / 2 . 9 \times 1 0 ^ { - 4 }$ </td><td> $2 . 0 \times 1 0 ^ { - 3 } / 2 . 1 \times 1 0 ^ { - 3 } / 4 . 2 \times 1 0 ^ { - 4 }$ </td><td> $2 . 4 \times 1 0 ^ { - 3 } / 2 . 5 \times 1 0 ^ { - 3 } / 3 . 5 \times 1 0 ^ { - 4 }$ </td><td> $2 . 5 \times 1 0 ^ { - 3 } / 2 . 5 \times 1 0 ^ { - 3 } / 3 . 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>10 10</td><td>0.2</td><td> $5 . 9 \times 1 0 ^ { - 2 } / 6 . 2 \times 1 0 ^ { - 2 } / 2 . 3 \times 1 0 ^ { - 3 } \ :$   $1 . 4 \times 1 0 ^ { - 1 } / 1 . 5 \times 1 0 ^ { - 1 } / 3 . 2 \times 1 0 ^ { - 3 }$ </td><td> $4 . O \times 1 0 ^ { - 2 } / 4 . I \times I O ^ { - 2 } / 2 . 3 \times 1 0 ^ { - 3 }$   $I . 2 \times I O ^ { - I } / I . I \times I O ^ { - I } / 6 . 4 \times 1 0 ^ { - 3 }$ </td><td> $2 . 8 \times 1 0 ^ { - 2 } / 2 . 9 \times 1 0 ^ { - 2 } / 2 . 4 \times 1 0 ^ { - 3 }$   $\mathbf { 8 . 2 \times 1 0 ^ { - 2 } } / 7 . 8 \times 1 0 ^ { - 2 } / 6 . 4 \times 1 0 ^ { - 3 }$ </td><td> $3 . 8 \times 1 0 ^ { - 2 } / 3 . 9 \times 1 0 ^ { - 2 } / 2 . 3 \times 1 0 ^ { - 3 }$ </td><td> $3 . 4 \times 1 0 ^ { - 2 } / 3 . 6 \times 1 0 ^ { - 2 } / 2 . 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>overall</td><td></td><td> $\overline { { 1 . 6 \times 1 0 ^ { - 1 } / 1 . 2 \times 1 0 ^ { - 1 } / 2 . 2 \times 1 0 ^ { - 2 } } }$ </td><td> $\overline { { 6 . 7 \times { I O ^ { - 2 } / 6 . 7 \times 1 0 ^ { - 2 } / 8 . 7 \times 1 0 ^ { - 4 } } } }$ </td><td> $\overline { { 6 . 1 \times 1 0 ^ { - 2 } / 1 . 4 \times 1 0 ^ { - 1 } / 4 . 1 \times 1 0 ^ { - 2 } } }$ </td><td> $1 . 0 \times 1 0 ^ { - 1 } / 1 . 0 \times 1 0 ^ { - 1 } / 4 . 6 \times 1 0 ^ { - 3 }$   $\overline { { 1 . 1 \times 1 0 ^ { - 2 } / 1 . 0 \times 1 0 ^ { - 2 } / 9 . 7 \times 1 0 ^ { - 4 } } }$ </td><td> $\mathbf { 9 . 4 \times 1 0 ^ { - 2 } } / 9 . 3 \times 1 0 ^ { - 2 } / 4 . 4 \times 1 0 ^ { - 3 }$   $\overline { { 1 . 0 \times 1 0 ^ { - 2 } / 9 . 6 \times 1 0 ^ { - 3 } / 8 . 8 \times 1 0 ^ { - 4 } } }$ </td></tr></table>

## Appendix B. Ablation studies

In ghost tasking, but also in PIGP for homogeneous equations, we need to condition on tasks that we know are exactly zero everywhere. We want to know if adding extra collocation points to condition on for the zero-valued tasks improve the results compared to the "standard" variant of having the same amount of points in each task. Additionally, since we know the collocation points to be exactly zero, we choose to constrain collocation points in exactly known tasks to a small noise nugget of $\sigma = 1 0 ^ { - 3 }$ . We refer to this method with extra, exactly known collocation points with a $" + "$ (e.g. "GT+" and "PIGP+"). We show the median of the errors of the learned MAP estimates of the physical parameters over the 10 runs relative to their true values in the inverse settings of all experiments in Tab. B.2-B.3 for the standard and the "+" variants. The variant that performed best across methods is marked with bold, and the best variant within a method is marked in italics. In the last rows, we show the overall performance calculated by the geometric mean of the table entries.

In our first experiment (chapter 4.1), "GT+" shows strong improvements especially for the ratio of the parameters $g / \ell ,$ which is strongly related to the eigenfrequency of the system. As discussed, this is likely due to the original GT method absorbing physically relevant oscillations in the ghost task, which can be suppressed with "GT+". "PIGP+" cannot be defined, since per construction, we measure its inputs $f _ { i } ( t )$ alongside the other tasks and can therefore not provide arbitrarily many collocation points as in GT. This is an inherent advantage of GT, but for the sake of the ablation studies, we consider a new variant which we will call "known $f ( t ) "$ We assume the inhomogeneity $f ( t )$ is fully known and can therefore be conditioned on as many extra collocation points as the ghost task. In this case, we condition both $\xi _ { G T }$ and $f ( t )$ for GT, and $f _ { 1 } ( t ) , f _ { 2 } ( t )$ , and $f _ { 3 } ( t )$ for PIGP on 25 noiseless input points. In this limit, GT and PIGP perform similarly and we recover the multimodal posterior distribution of l and g also with PIGP (Fig. B.14. For the other two experiments, adding extra collocation points does not lead to similarly strong improvements.

Table B.2: Ablation study experiment 2. Median absolute error of the physical parameter α relative to its true value $\alpha _ { t r u e } = 2 ,$ analogously to B.1. For the $" + "$ variant, the ghost task resp. the inhomogeneities (for GT resp. PIGP) are constrained on 15× 15 points and the standard deviation of the likelihood in these tasks is fixed to be σ = 0.001. Contrary to experiment 1 (B.1), we do not see significant differences between variants in the results.
<table><tr><td colspan="2">∆α α</td><td colspan="4">GT (torsion-free)</td><td colspan="4">GT (projective)</td><td colspan="4">PIGP</td></tr><tr><td>n</td><td>σ</td><td colspan="2">standard</td><td></td><td>十</td><td colspan="2">standard</td><td colspan="2">+</td><td colspan="2">standard</td><td colspan="2">+</td></tr><tr><td>3</td><td>0.001</td><td></td><td> $2 . 5 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> $6 . 6 \ \times I O ^ { - 4 }$ </td><td></td><td> $7 . 6 \times 1 0 ^ { - 3 }$ </td><td></td><td> $5 . I \ \times I O ^ { - 4 }$ </td><td></td><td> $5 . 8 \ \times 1 0 ^ { - 3 }$ </td><td></td><td> $3 . 6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>3</td><td>0.01</td><td></td><td> $2 . 5 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> $\mathbf { 3 . 1 \times 1 0 ^ { - 3 } }$ </td><td></td><td> $I . O \ \times I O ^ { - 2 }$ </td><td></td><td> $1 . 3 \ \times 1 0 ^ { - 2 }$ </td><td></td><td>8.7 × 10−-3</td><td></td><td> $3 . 4 ~ \times I O ^ { - 3 }$ </td></tr><tr><td>3</td><td>0.1</td><td></td><td> $3 . 6 ~ \times I O ^ { - 2 }$ </td><td></td><td> $4 . 4 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> $3 . 3 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> $8 . 3 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> $4 . 4 \ \times I O ^ { - 2 }$ </td><td></td><td> $1 . 1 \ \times 1 0 ^ { - 1 }$ </td></tr><tr><td>3</td><td>0.2</td><td></td><td> $9 . 5 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> $2 . 6 \times 1 0 ^ { - 1 }$ </td><td></td><td> $I . 7 ~ \times I O ^ { - I }$ </td><td></td><td> $7 . 3 \times 1 0 ^ { - 1 }$ </td><td></td><td> $I . I \ \times I O ^ { - I }$ </td><td></td><td> $6 . 4 ~ \times 1 0 ^ { - 1 }$ </td></tr><tr><td>5</td><td>0.001</td><td></td><td> $4 . 8 \ \times 1 0 ^ { - 4 }$ </td><td></td><td> $I . 9 \ \times I O ^ { - 4 }$ </td><td></td><td> $1 . 5 \ \times 1 0 ^ { - 4 }$ </td><td></td><td> $\mathbf { 1 . 4 \times 1 0 ^ { - 4 } }$ </td><td></td><td> $3 . 5 \ \times 1 0 ^ { - 4 }$ </td><td></td><td> $I . 9 \ \times I O ^ { - 4 }$ </td></tr><tr><td>5</td><td>0.01</td><td></td><td> $1 . 8 \ \times 1 0 ^ { - 3 }$ </td><td></td><td> $\mathbf { 1 . 7 \times 1 0 ^ { - 3 } }$ </td><td></td><td> $2 . 3 \times 1 0 ^ { - 3 }$ </td><td></td><td> $I . 9 \ \times I O ^ { - 3 }$ </td><td></td><td> $I . 8 \ \times I O ^ { - 3 }$ </td><td></td><td> $1 . 8 \ \times 1 0 ^ { - 3 }$ </td></tr><tr><td>5</td><td>0.1</td><td></td><td> $I . 8 \ \times I O ^ { - 2 }$ </td><td></td><td> $1 . 9 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> $I . 9 \ \times I O ^ { - 2 }$ </td><td></td><td> $3 . 2 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> $1 . 3 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> $2 . 6 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>5</td><td>0.2</td><td></td><td> $1 . 2 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> $5 . 7 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> $4 . 4 \ \times I O ^ { - 2 }$ </td><td></td><td> $1 . 8 \ \times 1 0 ^ { - 1 }$ </td><td></td><td> $3 . I \ \times I O ^ { - 2 }$ </td><td></td><td> $3 . 0 ~ \times 1 0 ^ { - 1 }$ </td></tr><tr><td>10</td><td>0.001</td><td></td><td> $9 . 0 ~ \times I O ^ { - 5 }$ </td><td></td><td> $1 . 0 \ \times 1 0 ^ { - 4 }$ </td><td></td><td> $8 . 7 ~ \times 1 0 ^ { - 5 }$ </td><td></td><td> $\mathbf { 8 . 7 \times 1 0 ^ { - 5 } }$ </td><td></td><td> $9 . 3 \ \times 1 0 ^ { - 5 }$ </td><td></td><td> $9 . 0 ~ \times I O ^ { - 5 }$ </td></tr><tr><td>10</td><td>0.01</td><td></td><td> $6 . 1 \ \times 1 0 ^ { - 4 }$ </td><td></td><td> ${ \bf 4 . 7 \times 1 0 ^ { - 4 } }$ </td><td></td><td> $1 . 2 \times 1 0 ^ { - 3 }$ </td><td></td><td> $I . 2 \ \times I O ^ { - 3 }$ </td><td></td><td> $1 . 2 \ \times 1 0 ^ { - 3 }$ </td><td></td><td> $I . 2 \ \times I O ^ { - 3 }$ </td></tr><tr><td>10</td><td>0.1</td><td></td><td> $1 . 2 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> ${ \bf 6 . 3 \ x 1 0 ^ { - 3 } }$ </td><td></td><td> $1 . 1 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> $9 . 0 ~ \times I O ^ { - 3 }$ </td><td></td><td> $1 . 0 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> $I . O \ \times I O ^ { - 2 }$ </td></tr><tr><td>10</td><td>0.2</td><td></td><td> $2 . 4 \times 1 0 ^ { - 2 }$ </td><td></td><td> $2 . 5 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> $4 . 7 \ \times I O ^ { - 2 }$ </td><td></td><td> $7 . 0 \times 1 0 ^ { - 2 }$ </td><td></td><td> $3 . 9 \ \times I O ^ { - 2 }$ </td><td></td><td> $7 . 8 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>overall</td><td></td><td></td><td> $6 . 6 \ \times 1 0 ^ { - 3 }$ </td><td></td><td> $4 . 4 \ \times 1 0 ^ { - 3 }$ </td><td></td><td> $6 . 6 \ \times I O ^ { - 3 }$ </td><td></td><td> $7 . 6 \times 1 0 ^ { - 3 }$ </td><td></td><td> $6 . I \ \times I O ^ { - 3 }$ </td><td></td><td> $7 . 2 \times 1 0 ^ { - 3 }$ </td></tr></table>

Table B.3: Ablation study experiment 3. Median absolute error of the physical parameter v0 relative to its true value $\nu _ { 0 t r u e } = 1$ , analogously to B.1. For the $" + "$ variant, the ghost task resp. the inhomogeneities (for GT resp. PIGP) are constrained on 10 × 10 × 10 points and the standard deviation of the likelihood in these tasks is fixed to be $\sigma = 0 . 0 0 1$ GT outperforms PIGP in every setting. Contrary to experiment 1 (B.1), we do not see significant differences between variants in the results.
<table><tr><td rowspan="2">∆v0 V0 n</td><td rowspan="2">σ</td><td colspan="3">GT</td><td colspan="3">PIGP</td></tr><tr><td>standard</td><td></td><td>+</td><td>standard</td><td></td><td>+</td></tr><tr><td>3</td><td>0.001</td><td> $1 . 0 ~ \times 1 0 ^ { - 1 }$ </td><td></td><td> $\bf 8 . 1 _ { \mathrm { ~ \scriptsize ~ \times ~ } } \bf 1 0 ^ { - 2 }$ </td><td> $3 . 4 ~ \times I O ^ { - I }$ </td><td>4.8</td><td> $\times 1 0 ^ { - 1 }$ </td></tr><tr><td>3</td><td>0.01</td><td> $9 . 6 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> ${ \bf 7 . 3 _ { \mathrm { ~ \scriptsize ~ \times ~ } } } \bf { 1 0 ^ { - 2 } }$ </td><td> $3 . 5 \ \times I O ^ { - I }$ </td><td></td><td> $4 . 9 \ \times 1 0 ^ { - 1 }$ </td></tr><tr><td>3</td><td>0.1</td><td> $8 . 3 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> ${ \bf 5 . 4 } ~ \times { \bf 1 0 ^ { - 2 } }$ </td><td> $3 . 9 \ \times I O ^ { - I }$ </td><td>5.3</td><td> $\times 1 0 ^ { - 1 }$ </td></tr><tr><td>3</td><td>0.2</td><td> $7 . 8 \times 1 0 ^ { - 2 }$ </td><td></td><td> $4 . 7 \ \times 1 0 ^ { - 2 }$ </td><td> $4 . 3 \ \times I O ^ { - I }$ </td><td></td><td> $5 . 5 \ \times 1 0 ^ { - 1 }$ </td></tr><tr><td>5</td><td>0.001</td><td> $9 . 4 \ \times 1 0 ^ { - 4 }$ </td><td></td><td> $\mathbf { 8 . 8 \times 1 0 ^ { - 4 } }$ </td><td> $3 . 2 \ \times I O ^ { - I }$ </td><td></td><td> $3 . 8 \ \times 1 0 ^ { - 1 }$ </td></tr><tr><td>5</td><td>0.01</td><td> ${ \bf 7 . 2 \ x 1 0 ^ { - 4 } }$ </td><td>8.4</td><td> $\times 1 0 ^ { - 4 }$ </td><td> $3 . 6 ~ \times 1 0 ^ { - 1 }$ </td><td>2.8</td><td> $\times 1 0 ^ { - I }$ </td></tr><tr><td>5</td><td>0.1</td><td> $3 . 8 \ \times 1 0 ^ { - 3 }$ </td><td></td><td> $3 . 3 \ \times 1 0 ^ { - 3 }$ </td><td> $3 . 7 \ \times 1 0 ^ { - 1 }$ </td><td></td><td> $I . 9 \ \times I O ^ { - I }$ </td></tr><tr><td>5</td><td>0.2</td><td> $5 . 0 \ \times 1 0 ^ { - 3 }$ </td><td></td><td> $4 . 7 \ \times 1 0 ^ { - 3 }$ </td><td> $3 . 7 \ \times 1 0 ^ { - 1 }$ </td><td></td><td> $I . 8 \ \times I O ^ { - I }$ </td></tr><tr><td>7</td><td>0.001</td><td> $1 . 2 \ \times 1 0 ^ { - 2 }$ </td><td></td><td> $1 . 2 \ \times 1 0 ^ { - 2 }$ </td><td> $1 . 1 \ \times 1 0 ^ { - 1 }$ </td><td></td><td> $I . I \ \times I O ^ { - I }$ </td></tr><tr><td>7</td><td>0.01</td><td> $1 . 2 \ \times 1 0 ^ { - 4 }$ </td><td></td><td> $9 . 7 \ \times 1 0 ^ { - 5 }$ </td><td> $1 . 7 \times 1 0 ^ { - 1 }$ </td><td></td><td> $I . O \ \times I O ^ { - I }$ </td></tr><tr><td>7</td><td>0.1</td><td> $4 . 3 \ \times 1 0 ^ { - 4 }$ </td><td></td><td> $4 . 2 \ \times 1 0 ^ { - 4 }$ </td><td> $3 . 1 \ \times 1 0 ^ { - 1 }$ </td><td></td><td> $2 . l ~ \times l { 0 } ^ { - I }$ </td></tr><tr><td>7</td><td>0.2</td><td> $\mathbf { 1 . 1 \times 1 0 ^ { - 3 } }$ </td><td></td><td> $1 . 2 \times 1 0 ^ { - 3 }$ </td><td> $3 . 4 ~ \times 1 0 ^ { - 1 }$ </td><td></td><td> $I . 6 ~ \times I O ^ { - I }$ </td></tr><tr><td colspan="2">overall</td><td> $5 . 3 \ \times 1 0 ^ { - 3 }$ </td><td></td><td> $4 . 6 \times 1 0 ^ { - 3 }$ </td><td> $3 . 0 ~ \times 1 0 ^ { - 1 }$ </td><td></td><td> $2 . 6 ~ \times I O ^ { - I }$ </td></tr></table>

![](images/293e58719a6d2c310b82cf4b4ec786ecdd3849abf29655e1c29cf51cf252c1bc.jpg)  
Figure B.14: Re-scaled log posteriors of $g , \ell$ analogously to Fig. 6 assuming $f ( t )$ is known. We recover similar multimodal structures as we did in the original experiment, but this time for both methods.

## Appendix C. Application to systems with general non-constant coefficients

We can find parametrizations even for certain DEs with non-constant coefficents via commutative polynomial rings. Such commuting cases can be found for instance in Example 4.2 of [10] and Example 1 of [25], the latter of which we shall use as an illustrative example. If we try to find a parametrization for the system represented by

$$
R ( { \bf x } ) = \left( \begin{array} { c c c } { { \nabla ^ { 2 } + g _ { 1 } ( { \bf x } ) } } & { { g _ { 2 } ( { \bf x } ) } } & { { - 1 } } \\ { { g _ { 3 } ( { \bf x } ) } } & { { \nabla ^ { 2 } + g _ { 4 } ( { \bf x } ) } } & { { 0 } } \end{array} \right) .\tag{C.1}
$$

with general functions $g _ { i } ( \mathbf { x } ) = g _ { i } ( x _ { 1 } , x _ { 2 } )$ , we could do so by assuming $g _ { i } ( \mathbf { x } )$ as (implicitly constant) parameters and using the Macaulay2 algorithm over commutative rings as in [10], and obtain a parametrization of the form

$$
\begin{array} { r } { Q ( \mathbf { x } ) = \left( \begin{array} { c } { g _ { 4 } ( \mathbf { x } ) + \nabla ^ { 2 } } \\ { - g _ { 3 } ( \mathbf { x } ) } \\ { - g _ { 2 } ( \mathbf { x } ) g _ { 3 } ( \mathbf { x } ) + g _ { 1 } ( \mathbf { x } ) g _ { 4 } ( \mathbf { x } ) + \nabla ^ { 2 } \big ( g _ { 1 } ( \mathbf { x } ) + g _ { 4 } ( \mathbf { x } ) \big ) + ( \nabla ^ { 2 } ) ^ { 2 } } \end{array} \right) . } \end{array}\tag{C.2}
$$

Evaluating R(x)Q(x) yields

$$
R ( { \bf x } ) Q ( { \bf x } ) = { \binom { \nabla ^ { 2 } g _ { 4 } ( { \bf x } ) - g _ { 4 } ( { \bf x } ) \nabla ^ { 2 } } { - \nabla ^ { 2 } g _ { 3 } ( { \bf x } ) + g _ { 3 } ( { \bf x } ) \nabla ^ { 2 } } } \doteq { \binom { 0 } { 0 } } ,\tag{C.3}
$$

from which we deduce that the parametrization is valid precisely when $g _ { 3 } ( \mathbf { x } )$ and $g _ { 4 } ( \mathbf { x } )$ commute with $\nabla ^ { 2 }$ , i.e., they must be constants. Notably, no such commutativity constraints apply to $g _ { 1 } ( \mathbf { x } )$ and $g _ { 2 } ( \mathbf { x } )$

This commutativity requirement is a strong restriction: in general applications, $g _ { 3 } ( \mathbf { x } )$ and $g _ { 4 } ( \mathbf { x } )$ will depend on their arguments, and thus the commutative parametrization is not available. Nonetheless, when the elements of $R ( \mathbf { x } )$ commute with the elements of $Q ( \mathbf { x } )$ , this approach provides a convenient and computationally inexpensive way to obtain a parametrization using standard tools for commutative polynomial rings.