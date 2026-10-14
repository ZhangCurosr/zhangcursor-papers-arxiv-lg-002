# Operator Surrogates

Christoph Schwab<sup>1,a</sup>, Josephine Westermann<sup>2,1,b</sup>, and Jakob Zech<sup>2,c</sup>

<sup>1</sup>SAM, ETH Zürich, Rämistrasse 101, 8092 Zürich, Switzerland   
<sup>2</sup>IWR, Uni Heidelberg, INF 205, 69120 Heidelberg, Germany <sup>a</sup>schwab@math.ethz.ch <sup>b</sup>josephine.westermann@math.ethz.ch <sup>c</sup>jakob.zech@uni-heidelberg.de

July 15, 2026

## Abstract

This chapter provides a mathematical introduction to operator surrogates, which are finite-parametric maps approximating complex (typically nonlinear) operators between infinitedimensional (function) spaces.

A typical approach to constructing operator surrogates is the encoder-approximator-decoder paradigm, whereby infinite-dimensional inputs are mapped to a finite-dimensional latent space, processed, and then mapped back to the infinite-dimensional output space. We discuss various methods for creating eficient representation systems, including stable linear frameworks like Banach frames and data-driven techniques such as Principal Component Analysis (PCA), Active Subspaces, and nonlinear autoencoders. Subsequently, we survey key surrogate architectures, contrasting spectral methods, based on polynomial chaos expansions and related technologies, with more recent neural network approaches. We further discuss architectures that do not rely on representation systems, such as Fourier Neural Operators and transformers. We elaborate on their relationship and how they relate to other neural network operator surrogates, particularly those obtained by so-called ‘algorithm-unrolling’.

Additionally, we provide an introduction to the theoretical analysis of approximation and generalization errors. We explain how underlying low-dimensional structures, provided for example by holomorphy, are essential to overcome the so-called “curse of dimensionality”: this refers to the fact that the complexity of numerical algorithms tends to increase exponentially in the problem dimensionality. Since operator surrogates deal with the approximation of mappings between infinite dimensional spaces, traditional approaches suitable for low-dimensional problems typically fail in these situations. We present convergence rate bounds for holomorphic operators that depend on the number of trainable parameters and the regularity of the input and output data. Throughout, we prioritize intuition over detailed technical proofs and provide precise pointers to the literature for detailed arguments.

## Contents

1 Introduction 3   
1.1 Existing Results and Algorithms . . . 5   
1.2 Goals and Audience . 6   
1.3 Outline 6   
1.4 Code 6   
1.5 Notation 6   
2 Model Problems 7   
2.1 Stationary Difusion 7   
2.2 Finite-dimensional Initial Value Problems 9   
2.3 Parabolic Initial-Boundary Value Problems 13   
3 Representation Systems 14   
3.1 Stable linear representations 15   
3.2 PCA 18   
3.2.1 Formulation . 18   
3.2.2 Characterization of V 19   
3.2.3 Computation of Encoder and Decoder 20   
3.3 Active Subspaces 21   
3.3.1 Formulation . 22   
3.3.2 Characterization of V 23   
3.3.3 Computation of Encoder 24   
3.4 Autoencoders 25   
3.4.1 Afine Autoencoder . 25   
3.4.2 Nonlinear Autoencoder . 26   
3.4.3 Active Subspace Autoencoder 26   
3.4.4 Computation of Encoder and Decoder 26   
4 Architectures 27   
4.1 Spectral Operators 27   
4.1.1 Polynomial Chaos Expansions . 28   
4.1.2 Sparse-Grid Interpolation 29   
4.1.3 Low-rank Tensor Approximations 30   
4.2 Nonlocal Operator Networks 30   
4.3 PCA-Net and FrameNet 32   
4.4 DeepONet 32   
4.5 Transformers 33   
5 Training 35   
5.1 Empirical Risk Minimization 36   
5.2 Adding Derivatives 36   
6 Approximation Errors 37   
6.1 Universal Approximation . 37   
6.2 C<sup>k</sup> Operators . . 39   
6.3 Holomorphic Operators 42   
6.3.1 Smoothness Spaces 42   
6.3.2 Convergence Rates 42   
6.4 Algorithm unrolling 45

## 1 Introduction

Partial (Integro) Diferential Equations (P(I)DEs for short) are the most widely employed formalism to express continuum models in engineering and in the sciences. P(I)DEs do not admit explicit solutions, as a rule, and numerical approximations need to be employed for predictive simulations. Accordingly, during the past decades, numerical solution technology for large classes of PDEs has matured, with broad classes of solution methodologies (e.g. finite diference, finite element, finite volume, spectral methods ...) today being theoretically founded and practically deployed. The focus of this development was on eficient numerical approximation of PDE solutions for one instance of input data. In the course of this development, numerical PDE solvers have been optimized to in many cases near-linear complexity, in particular for elliptic and parabolic P(I)DEs.

Increasingly, however, the availability of ubiquitous observation and measurement data has generated new computational scenarios which mandate eficient numerical many-query PDE solutions. Examples include data assimilation [6], uncertainty quantification [137], digital twins (see [142, 5, 122, 97] and references there), eficient sampling of statistical solutions of evolution equations (e.g. [20, 52, 92] and the references there) etc., all of which mandate eficient many query solvers.

Parallel execution of the mentioned existing single query solution methodologies has been developed in the past decades but will, usually, not exploit common structure of input queries. This is well-known, for example, in Single-Level Monte-Carlo (SLMC) simulations in Uncertainty Quantification (UQ); as is by now well-known, Multi-Level discretizations of parametric PDEs such as Multi-Level Monte-Carlo (MLMC) simulation strategies ofer significant error-vs-work gains for PDEs with uncertain input data (e.g. [92]) albeit limited by the low MC convergence rate.

The eficient numerical realization of operators, that is generally nonlinear maps between infinitedimensional function spaces, has become a key research topic at the interface of scientific computing (SC) and machine learning (ML). A core task in this field, referred to as Scientific Machine Learning (SciML), is to provide eficient numerical surrogates $\hat { \mathcal G }$ for the data-to-solution maps

$$
\mathcal { G } : \mathcal { X }  \mathcal { Y }\tag{1.1}
$$

of operator equations, particularly PDEs; here  and $\mathcal { V }$ are typically Banach or Hilbert spaces. Operator surrogates are approximations of such mappings, where $\mathcal { G }$ is truly infinite-dimensional; by this we mean that the problem cannot be reduced to that of approximating a function whose domain (or codomain) is finite dimensional. We will make this more precise in Section 6.2. Such surrogates can be building blocks in algorithms that involve many-query evaluations of PDE models. Examples are inverse problems and algorithms such as MCMC or filtering for data assimilation into PDE models.

Mathematically, the goal is to construct parsimonious, finite-parametric approximations of these maps that come with rigorous a-priori performance guarantees. Ideally, these approximations should also achieve some form of optimality in terms of accuracy versus computational work for a given set of input data. This area has been significantly advanced by deep learning methodologies that assimilate data-solution pairs into operator surrogates. For recent overviews, we refer for example to the surveys [79, 81, 40] and the references therein.

With the currently emerging availability of hierarchic classes of finite-parametric maps between function spaces, which this chapter discusses, the area of data assimilation and data-driven updates in hierarchies of operator surrogates is emerging, in particular in connection with digital twins [142, 5, 6]. This line of reasoning tacitly assumes that the operator equations which describe the physics of interest are known in closed form.

Operator Learning refers to computational methodologies that associate, for an input-output relation given e.g. as solution map of an operator equation in function spaces (hence, with infinitedimensional domain and codomain), finite-parametric operator surrogates take the form of numerical approximations of such relations, ideally with performance guarantees. In this process, operator surrogates from a given set of surrogate architectures are numerically calibrated to the input-output relation, usually subject to data.

Within the context of AI and natural language processing, the process of operator learning and model updating itself can be transferred to computational methods [69, 109]. This is of particular interest for multi-modal data. The mathematical formalization of operator-learning methodologies in this context is still in its infancy and beyond the scope of this (introductory) chapter. However, the mathematical tools presented here will almost certainly be required in some form towards formalizing such algorithmic approaches.

Besides neural networks, operator surrogates can be built from other approximation architectures, and we provide a first example for dynamical systems in Section 2.2. Owing to the infinite dimension of the input and output function spaces, in the design and mathematical analysis of numerical operator approximation architectures, in particular in the establishment of convergence rate bounds, the so-called curse of dimensionality (CoD) takes a prominent role: approximation architectures which are successful in operator learning must realize some mechanism of overcoming the CoD. To date, several broad classes of approximation architectures have emerged with good performance on practical operator learning tasks and with mathematical foundations in various stages of completeness. A rough taxonomy could be:

(i) Spectral Operators based on (Generalized) Polynomial Chaos (g)PC Approximations and related methods: the CoD in the parametric approximations is overcome by sparsity of gPC expansions [71] in holomorphic operators. Operator classes of finite regularity (such as k-times Fréchet-diferentiable maps with finite k) generally sufer from the curse of dimensionality, see e.g. [80] and the references there.

(ii) Neural Operators, Neural Operator Networks: the CoD in the parametric approximations is overcome by sparsity in the operator and strong expressivity of deep neural networks, which allow us to detect such sparsity. For input and outputs from (infinite-dimensional) function spaces, this can be realized in the so-called encoder-approximator-decoder framework [87, 79, 95, 15, 53, 126].

(iii) Random Feature Neural Networks, Extreme Learning Machines and Reservoirs: the CoD is overcome by probabilistic architectures and error measures, and mathematical analysis furnishes statistical convergence guarantees without the CoD, via suitable central limit theorems [88, 58, 44].

(iv) Algorithm Unrolling: A further approach towards construction of operator networks can be termed “algorithm unrolling”. Going back to [110], it consists in re-writing numerical algorithms with mathematical convergence guarantees in terms of error vs. work for numerical solution approximation of nonlinear (operator) equations in terms of deep neural network architectures. One execution of an iterative solver (e.g. [22]) respectively of a constructive proof of existence of solutions (e.g. [51]), will give rise to a map of the solution (approximation) space into itself, and will entail recurrent or so-called “U-Net” structures within the Operator Network architecture.

(v) Kernel Methods: Kernel regression has in recent years been proposed for operator learning e.g. in [11, 23, 99].

We shall focus in the present review on details for methodologies i) and ii), but do not develop technical details of approaches iii) - v). Any of the above can be combined by composition.

## 1.1 Existing Results and Algorithms

Fundamental mathematical questions about classes of neural operators are roughly grouped around approximation properties on the one hand, and computational feasibility and eficiency (and, related to this, generalization, optimization, robustness/stability and sample complexity, to name but a few) on the other.

Approximation properties describe whether a given class of neural operators can, as the number of trainable parameters tends to infinity, in principle approximate a given map between function spaces to any prescribed tolerance, in a given metric and for a given class of inputs. This is commonly referred to as universal approximation. Universal approximation of neural operator surrogates has been established for a wide range of architectures. We mention only the seminal work [21], which focused on uniform (“worst case”) universality on compact subsets of inputs, and subsequent work [87, 71, 133] where compactness of admissible inputs was relaxed, and $L ^ { p } ( { \mathcal { X } } , { \mathcal { Y } } ; \mu )$ - universality was shown. We also mention the recent [55, 12, 57] where universal approximation was shown for a wide range of architectures comprising those considered here, in particular transformers, acting between Banach-, Metric- and Fréchet-spaces.

A more refined viewpoint asks at which rates can a given class of neural operator surrogates approximate suitable maps $\mathcal { G }$ between function spaces, given conditions on the set of admissible input data. Mathematical results on operator approximation rates are, at this time, less developed than results on universal approximation. They require specific assumptions on admissible input sets, typically related to regularity in scales $\{ \mathcal { X } ^ { s } \} _ { s > 0 }$ of subspaces ${ \mathcal { X } } ^ { s } \subset { \mathcal { X } }$ of the input space, and stronger structural assumptions.

Mathematical results on approximation rates typically assume exact representation (i.e. infiniteprecision) of network weights and biases. In this idealized scenario, the Kolmogorov-Arnold representation theorem [77] enables universality and approximation rates with shallow NNs (see, e.g., [133] for concrete results). When implemented on digital computers with finite-precision, however, the Kolmogorov construction faces stability obstructions. For a mathematical analysis of quantization in the context of function approximation see [121] and references there. For leveraging quantization in neural operators see [145]. Further, in [84, 85] operator network expressivity is studied with focus on information-theoretic aspects, where the finite bitstring length of floating point numbers is taken into account.

## 1.2 Goals and Audience

This chapter introduces key mathematical concepts and algorithmic foundations of the growing field of operator surrogates. Our primary goal is to build intuition through a selective overview of the field. Nonetheless, we provide mathematically precise statements, and give proofs or proof sketches of some key results. For more intricate and technical arguments, readers are directed to the relevant literature. The material is aimed at a Master’s-level audience and assumes familiarity with basic concepts in finite-dimensional deep learning, as, e.g., presented in [123].

## 1.3 Outline

We begin in Section 2 by introducing two prototypical model problems to illustrate basic concepts and applications of operator learning. In Section 3 we introduce tools for eficient dimension reduction and representation of elements in the input and output spaces $\mathcal { X }$ and . This is a crucial prerequisite for constructing any numerical operator surrogate of $\mathcal { G }$ in (1.1).

With these fundamentals in place, we move to the constructive and analytical core: Section 4 discusses and compares a selection of surrogate architectures, with a focus on spectral and neural operators. The practical task of training such models is explained in Section 5, where we detail a typical training setup for an encoder-decoder architecture. Next, in Section 6 we transition to a theoretical analysis, and discuss approximation properties of these surrogates. We start by reviewing the relevance of compactness and its implications for universal approximation. Afterwards we explain the necessary occurrence of the curse of dimension for operator classes of finite regularity. While there are diferent scenarios under which it can be overcome, we focus on the class of holomorphic operators. Building on these results, in Section 7 we briefly discuss the topic of generalization, without going into details on the proofs.

Finally, Section 8 concludes the chapter by providing an outlook on extensions of these results, discussing methods not covered, and pointing to further relevant literature.

## 1.4 Code

This work is accompanied by a repository of Python Jupyter notebooks, available at github.com/ JoWestermann/operator-surrogates/tree/release/v1, implementing core concepts from the text alongside visualizations and numerical experiments. Throughout the manuscript, text boxes marked with the icon - point the reader to specific notebooks related to the discussed concepts.

## 1.5 Notation

Numbers and constants. We write $\mathbb { N } = \{ 1 , 2 , \ldots \}$ and $\mathbb { N } _ { 0 } = \{ 0 , 1 , 2 , \ldots \}$ . Throughout, $C \lesssim D$ means that $C$ can be bounded by a multiple of D, independently of parameters which $C$ and D may depend on. $C \gtrsim D$ is defined as $D \lesssim C$ , and $C \simeq D$ as $C \lesssim D$ and $C \gtrsim D$

Banach and Hilbert spaces. For a Hilbert or Banach space we denote its norm by $\| \cdot \| _ { \mathcal { X } }$ . For a Hilbert space $x ,$ , the inner product of v and $w \in \mathcal { X }$ is denoted by $\langle v , w \rangle$ . The (unique, cf., $\mathrm { e . g . }$ [111]) complexification of a real Hilbert space is denoted by $\mathcal { X } _ { \mathbb { C } }$ . Similarly, for Banach spaces $\mathcal { X }$ we let $\mathcal { X } _ { \mathbb { C } }$ be a complexification with a natural norm [111, 156].

For Banach spaces  and $\mathcal { V } _ { : }$ , we denote by $L ( \mathcal { X } , \mathcal { Y } )$ the bounded, linear maps from $\mathcal { X }$ to $\mathcal { V } ,$ by $C ( \mathcal X , \mathcal Y )$ the continuous maps, and for $k \in  { \mathbb { N } } _ { 0 }$ by $C ^ { k } ( \mathcal { X } , \mathcal { Y } )$ the k-times continuously Fréchet diferentiable maps. The kth Fréchet derivative of $\mathcal { G } \in C ^ { k } ( \mathcal { X } , \mathcal { Y } )$ at $x \in \mathcal { X }$ is denoted by $D ^ { k } { \mathcal { G } } ( x ) \in$ $L ( \mathcal { X } , \mathcal { Y } )$ , and if is a Hilbert space and $\mathcal { V } = \mathbb { R }$ , then $\boldsymbol { \nabla } \mathcal { G } ( \boldsymbol { x } ) \in \mathcal { X }$ is the gradient. Moreover, $\mathcal { X } ^ { \prime } = L ( \mathcal { X } , \mathbb { R } )$ is the dual space of .

Lebesgue and Sobolev spaces. For $p \in ( 0 , \infty )$ , the space of real-valued, p-summable sequences indexed over N is denoted by $\ell ^ { p } ( \mathbb { N } )$ . All Lebesgue spaces with summability exponent p are assumed to have the usual modification for $p = \infty$

For a Lipschitz domain $\mathbf { D } \subseteq \mathbb { R } ^ { n }$ , we use the usual function spaces: $L ^ { 1 } ( \mathbb { D } )$ denotes the space of absolutely (Lebesgue) integrable, real-valued functions in D. For $1 ~ < ~ p ~ < ~ \infty$ , we denote $L ^ { p } ( \mathrm { D } ) : = \{ u : \mathrm { D } \to \mathbb { R } , | u | ^ { p } \in L ^ { 1 } ( \mathrm { D } ) \}$ the space of p-Lebesgue integrable, real-valued functions, with corresponding norm denoted by $\| u \| _ { L ^ { p } ( \mathrm { D } ) }$ . Again, for $p = \infty$ the usual modification. For $k \in \mathbb N$ we let $H ^ { k } ( \mathrm { D } ) = \{ f : \mathrm { D } \to \mathbb { R } : \partial ^ { \nu } f \in L ^ { 2 } ( \mathrm { D } ) \ \forall \pmb { \nu } \in \mathbb { N } _ { 0 } ^ { d } : | \pmb { \nu } | \leq k \}$ . Finally, $\mathrm { t r } \in L ( H ^ { 1 } ( \mathrm { D } ) , H ^ { 1 / 2 } ( \partial \mathrm { D } ) )$ is the trace map, $H _ { 0 } ^ { 1 } ( \mathrm { D } ) : = \{ u \in H ^ { 1 } ( \mathrm { D } ) : \operatorname { t r } ( u ) = 0 \}$ , and $H ^ { - 1 } ( \mathrm { { D } ) }$ is the dual space of $H _ { 0 } ^ { 1 } ( \mathrm { D } )$

## 2 Model Problems

We illustrate basic concepts of operator learning for two model problems. The first is the standard parametric elliptic PDE, and we review known results on well-posedness and parameter regularity. The second is the flow of an ODE in finite dimensions; while technically not operator learning (due to the finite dimensionality), it allows us to showcase key features of evolution problems.

## 2.1 Stationary Difusion

Let $n \geq 2$ , and let $\mathbf { D } \subseteq \mathbb { R } ^ { n }$ be a bounded Lipschitz domain. Let $a \mathrm { ~ \textbar ~ { ~ } ~ } > 0$ , let a difusion coeficient a be given in

$$
L _ { a _ { - } } ^ { \infty } ( \mathrm { D } ) : = \left\{ a \in L ^ { \infty } ( \mathrm { D } ) : \displaystyle \exp \operatorname* { i n f } _ { \eta \in \mathrm { D } } a ( \eta ) \ge a _ { - } \right\} .\tag{2.1}
$$

With this coeficient and a source term $f \in L ^ { 2 } ( \mathrm { D } )$ , consider the boundary value problem (BVP): find $y \in H _ { 0 } ^ { 1 } ( \mathrm { D } )$ such that

$$
- \nabla _ { \eta } \cdot ( a ( \eta ) \nabla _ { \eta } y ( \eta ) ) = f \mathrm { i n } \mathrm { D } , \quad \mathrm { t r } ( y ) = 0 \mathrm { o n } \partial \mathrm { D } .\tag{2.2}
$$

To define weak solutions and discretizations, we adopt a variational formulation of (2.2). There are, generally, many possibilities to re-formulate (2.2). The most straightforward formulation seeks the solution $y \in \mathcal { Y } : = H _ { 0 } ^ { 1 } ( \mathrm { D } )$ and is obtained by multiplying (2.2) with a test-function $v \in H _ { 0 } ^ { 1 } ( \mathrm { D } )$ and integrating by parts. It reads: find $y \in H _ { 0 } ^ { 1 } ( \mathrm { D } )$ such that

$$
b ( a ; y , v ) : = \int _ { D } a ( \eta ) \nabla y ( \eta ) \cdot \nabla v ( \eta ) d \eta = \int _ { \mathrm { D } } f ( \eta ) v ( \eta ) d \eta \quad \forall v \in H _ { 0 } ^ { 1 } ( \mathrm { D } ) .\tag{2.3}
$$

We choose as data-space of problem (2.3) $\chi = L ^ { \infty } ( \mathrm { D } ) \times H ^ { - 1 } ( \mathrm { D } )$ , and endow it with the sum norm $\| ( f , a ) \| _ { \mathcal { X } } : = \| f \| _ { H ^ { - 1 } ( \mathrm { D } ) } + \| a \| _ { L ^ { \infty } ( \mathrm { D } ) }$

For every $a \in L _ { a _ { - } } ^ { \infty } ( \mathrm { D } )$ the bilinear form $b ( a ; \cdot , \cdot ) : \mathscr { V } \times \mathscr { V } \to \mathbb { R }$ is continuous and coercive, i.e. by (2.1)

$$
\begin{array} { r } { \forall v , w \in \mathcal { V } : \quad | b ( a ; v , w ) | \leq \| a \| _ { L ^ { \infty } ( \mathrm { D } ) } \| \nabla v \| _ { L ^ { 2 } ( \mathrm { D } ) } \| \nabla w \| _ { L ^ { 2 } ( \mathrm { D } ) } , \quad b ( a ; v , v ) \geq a _ { - } \| \nabla v \| _ { L ^ { 2 } ( \mathrm { D } ) } ^ { 2 } . } \end{array}\tag{2.4}
$$

By Poincaré’s inequality, the expression $\| \nabla v \| _ { L ^ { 2 } ( \mathrm { D } ) }$ is a norm on $\mathcal { V } = H _ { 0 } ^ { 1 } ( \mathrm { D } )$ , the so-called “energy norm”, i.e. $\begin{array} { r } { \| v \| _ { \mathcal { V } } ^ { 2 } \simeq \int _ { \mathrm { D } } a | \nabla v | ^ { 2 } d x = \| \sqrt { a } \nabla v \| _ { L ^ { 2 } ( \mathrm { D } ) } ^ { 2 } } \end{array}$ . Here, denotes equivalences with constants depending on a<sub>−</sub> in (2.1) and on an upper bound $a _ { + }$ of $\| a \| _ { L ^ { \infty } ( \mathrm { D } ) }$ Then, (2.4) implies with the Lax-Milgram Lemma, e.g., [46], that for every $f \in L ^ { 2 } ( \mathrm { D } )$ the BVP (2.2) admits a unique solution. Denote the solution operator by

$$
\mathcal { G } : = \{ \{ ( a , f ) \in \mathcal { X } : a \in L _ { a _ { - } } ^ { \infty } ( \mathrm { D } ) \}  \mathcal { Y }\tag{2.5}
$$

The map $\mathcal { G } ( . , . )$ thus defined is Lipschitz-continuous: for fixed data $f _ { 1 } , f _ { 2 } \in L ^ { 2 } ( \mathrm { D } )$ and for $a _ { 1 } , a _ { 2 } \in$ $L _ { a _ { - } } ^ { \infty } ( \mathrm { D } )$ , there exist constants $c _ { a } , c _ { f } > 0$ such that (see, e.g. [8])

$$
\begin{array} { r } { \| \mathcal { G } ( { a } _ { 1 } , f _ { 1 } ) - \mathcal { G } ( { a } _ { 2 } , f _ { 2 } ) \| _ { \mathcal { V } } \leq c _ { a } \| { a } _ { 1 } - { a } _ { 2 } \| _ { L ^ { \infty } ( \mathrm { D } ) } + c _ { f } \| f _ { 1 } - f _ { 2 } \| _ { L ^ { 2 } ( \mathrm { D } ) } . } \end{array}
$$

The solution locally even depends holomorphically on $( a , f ) \in { \mathcal { X } }$ , see, e.g., [29, 28]:

Remark 2.1. Holomorphic parameter dependence of y on a and f means that the solution operator $\mathcal { G } : ( a , f ) \mapsto y$ is complex Fréchet diferentiable as a mapping between the Banach spaces and in (2.5). It is a strong form of smoothness, that implies in particular infinite Fréchet (and Gâteaux) diferentiability. For example, letting $a ( t ) = a _ { 0 } + t a _ { 1 } , a _ { 0 } \in L _ { a _ { - } } ^ { \infty } ( \mathrm { D } )$ as in (2.1) and $\varphi \in \mathcal { V } ^ { \prime }$ , it holds that $t \mapsto \varphi ( \mathscr { G } ( a ( t ) , f ) ) : \mathbb { C } \to \mathbb { C }$ is complex diferentiable in a vicinity $o f \operatorname { 0 }$ , and thus is in particular $C ^ { \infty }$ in a neighbourhood of 0. This strong smoothness is a crucial property inherent to many (well-posed) PDEs, [29, 28].

Remark 2.2. Proving Neural Operator approximation rate bounds require, as we shall see below, mapping properties of  restricted to subspaces. Specifically, that on a suitable subspace ${ \tilde { \mathcal { X } } } \subset { \mathcal { X } } ,$ $\mathcal { G } \mid _ { \tilde { \mathcal { X } } }$ belongs to a corresponding subspace $\tilde { \mathcal { V } } \subset \mathcal { V }$ . Specific choices of $\tilde { \mathcal { X } } , \tilde { \mathcal { Y } }$ are problem-dependent. For (2.2) assume that D is convex, and consider data from $\tilde { \mathcal { X } } : = W ^ { 1 , \infty } ( \mathrm { D } ) \cap L _ { a _ { - } } ^ { \infty } ( \mathrm { D } ) \times L ^ { 2 } ( \mathrm { D } ) \subset \mathcal { X }$ $( i . e .$ Lipschitz-continuous in D difusion coeficient a and square-integrable source-term $f )$ . Then, by classical results on elliptic regularity,

$$
\mathcal { G } | _ { \tilde { \mathcal { X } } } \subset \tilde { \mathcal { Y } } : = ( H ^ { 2 } \cap H _ { 0 } ^ { 1 } ) ( \mathrm { D } ) .
$$

This regularity property will be in Section 6.3.2 below a key ingredient in the convergence rate analysis of en- and decoders in the architecture (3.1) in Section 3.

Theorem 2.3. Let $a _ { 0 } \in L _ { a _ { - } } ^ { \infty } ( \mathrm { D } )$ and $f _ { 0 } \in H ^ { - 1 } ( \mathrm { D } )$ . Then $\mathcal { G }$ in (2.5) is Fréchet diferentiable in a neighbourhood $o f \left( a _ { 0 } , f _ { 0 } \right)$ in the complexified spaces $L ^ { \infty } ( \mathrm { D } ) _ { \mathbb { C } } \times L ^ { 2 } ( \mathrm { D } ) _ { \mathbb { C } }$ (see [111] for complexification of real Banach spaces).

We only sketch the argument following [156, Example 1.2.38], and refer to this reference for further details: Since $\mathcal { G }$ is linear in $f _ { i }$ we focus on $a \mapsto \mathcal G ( a , f )$ . With the diferential operator

$$
A ( a ) : = \left\{ \begin{array} { l l } { H _ { 0 } ^ { 1 } ( \mathrm { D } ) \to H ^ { - 1 } ( \mathrm { D } ) } \\ { y \mapsto - \nabla \cdot ( a \nabla y ) } \end{array} \right.
$$

We have $A ( a _ { 0 } + a ) = A ( a _ { 0 } ) + A ( a )$ and $y ( a ) = A ( a ) ^ { - 1 } ( f ) = ( A ( a _ { 0 } ) + A ( a ) ) ^ { - 1 } f$ . By the stated unique weak solvability of (2.2), the map $A ( a _ { 0 } ) \in L ( H _ { 0 } ^ { 1 } ( \mathrm { D } ) , H ^ { - 1 } ( \mathrm { D } ) )$ is an isomorphism. Using a Neumann series we thus have

$$
y ( a _ { 0 } + a ) = ( A ( a _ { 0 } ) + A ( a ) ) ^ { - 1 } f = \sum _ { k \in \mathbb { N } _ { 0 } } ( - 1 ) ^ { k } ( A ( a _ { 0 } ) ^ { - 1 } A ( a ) ) ^ { k } A ( a _ { 0 } ) ^ { - 1 } f .
$$

The kth term in this expansion is a k-multilinear map in a. It thus represents a Taylor expansion in Banach spaces of the solution y around $a _ { 0 }$ . In particular the (symmetrized) kth term can, up to the constant k!, be identified with the kth (complex) Fréchet derivative of $y ( a _ { 0 } + a )$ with respect to a.

\- For an illustration of the parametric difusion problem for $ { n _ { \mathrm { ~  ~ } } } =  { \mathrm { ~ \ r ~ { ~ 1 ~ } ~ } }$ see notebook 0\_Diffusion.ipynb. The PDE solution is approximated via the finite diference method.

## 2.2 Finite-dimensional Initial Value Problems

Operator Networks have been proposed recently for learning solution maps of evolution (initialboundary value) problems. See, e.g. [9, 39] and the reference there. In these works, numerical “neural propagators”, i.e., operator networks which map e.g. initial data at time $t ~ = ~ 0$ to the solution at time $t > 0$ have been constructed by transferring a time-stepping PDE scheme into a suitable neural operator architecture, a procedure which has been also referred to as “algorithmunrolling” (see Section 6.4 ahead) a terminology coined in [110]. The operator-network concept allows, however, also fundamentally diferent approaches to evolution problems.

We illustrate this for an initial-value ODE system in finite-dimensional, Euclidean state space $\mathbb { R } ^ { n }$

Let the vector field $\Phi : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ be globally Lipschitz-continuous, i.e. there exists a constant $L _ { \Phi } > 0$ such that

$$
\forall u , v \in \mathbb { R } ^ { n } : ~ \| \Phi ( u ) - \Phi ( v ) \| _ { 2 } \leq L _ { \Phi } \| u - v \| _ { 2 } ~ .\tag{2.6}
$$

Given $u _ { 0 } \in \mathbb { R } ^ { n }$ and any time-horizon $0 < T \leq \infty .$ , consider the initial-value problem (IVP): find $u \in C ^ { 1 } ( [ 0 , T ] ; \mathbb { R } ^ { n } )$ such that for $0 < t \leq T$ holds

$$
\dot { u } ( t ) = \Phi ( u ( t ) ) \ , \quad u ( 0 ) = u _ { 0 } \ .\tag{2.7}
$$

It is classical (see, e.g., [149]) that under the global Lipschitz assumption (2.6), the IVP (2.7) admits a unique solution $( u ( t ) ) _ { t \geq 0 }$ , defined for all $t \geq 0$ . We remark that under weaker, local Lipschitz conditions on Φ, generally existence and uniqueness hold only up to a finite, maximal time-horizon T. All that follows holds verbatim for $0 < t < T$ in that case.

For fixed $t > 0$ , the initial-data-to-solution-map

$$
S ( t ) : \mathbb { R } ^ { n }  \mathbb { R } ^ { n } : u _ { 0 } \mapsto u ( t )\tag{2.8}
$$

is Lipschitz-continuous in the uniform norm [149, Chap.III.13.II]: for any compact set $K \subset \mathbb { R } ^ { n }$ of initial data, and for every $0 < T < \infty$ , there exists a constant $C ( K , T , L _ { \Phi } )$ such that

$$
\forall u _ { 0 } , \tilde { u } _ { 0 } \in K : \operatorname* { s u p } _ { 0 \leq t \leq T } \| u ( t ) - \tilde { u } ( t ) \| _ { 2 } = \operatorname* { s u p } _ { 0 < t \leq T } \| S ( t ) u _ { 0 } - S ( t ) \tilde { u } _ { 0 } \| _ { 2 } \leq C ( K , T ) \| u _ { 0 } - \tilde { u } _ { 0 } \| _ { 2 } .
$$

The global unique solvability of (2.7) gives that the nonlinear maps $S ( t )$ exhibit the (compositional) semigroup property

$$
S ( 0 ) = I , \quad \mathrm { a n d } \quad \forall t , t ^ { \prime } \geq 0 : \quad S ( t + t ^ { \prime } ) = S ( t ) \circ S ( t ^ { \prime } ) ~ .\tag{2.9}
$$

For vector fields Φ in (2.7) which have, in addition to (2.6), extra regularity, the map $S ( t )$ in (2.8) is more regular.

Theorem 2.4. Consider the ODE-IVP (2.7) in $\mathbb { R } ^ { n }$ . Assume that the vector field Φ is globally Lipschitz (2.6) and, in addition, that the vector field Φ is real-analytic on compact sets $K \subset \mathbb { R } ^ { n }$

Then, for every $0 \textless t \textless T$ , the initial-data to solution map $S ( t ) : u _ { 0 } \mapsto u ( t )$ for (2.7) is real-analytic, as a map from $K \subset \mathbb { R } ^ { n }$ to $S ( t ) ( K ) \subset \mathbb { R } ^ { n }$

Proof. The proof of this result follows from standard properties of ODE-IVPs (2.7) with (2.6): the Lipschitz-continuity of $S ( t )$ shown in [149, Chap.III.13.II] implies that $S ( t ) ( K )$ is compact. We next consider the notation in [149, Chap.III.12 & 13]: there, non-autonomous $\mathrm { O D E - I V P s } \dot { y } ( t ) : = { \pmb f } ( t , y , \lambda )$ are considered, with $\lambda \in K \subset \mathbb { R } ^ { n }$ , i.e. in a compact set of parameters. It is shown in [149, Chap.III.13, Thm. XI] that in the case that f is holomorphic in the variables $( t , y , \lambda )$ , so is the solution’s dependence on λ.

To reduce the proof of Theorem 2.4 to this result, we observe that our assumption of real analyticity of $f$ implies existence of a complex analytic extension of $f$ to a (possibly small) open neighborhood $U \subset \mathbb { C } ^ { n }$ which contains K, see, e.g., [91, Prop. 3.1.1]. We set for $u _ { 0 } \in K$ and $t \in [ 0 , T ] = : J , y ( t ) : = u ( t ) - u _ { 0 }$ , so that for all $\lambda = u _ { 0 } \in K$ holds $y ( 0 ) = \mathbf { 0 }$ , and $u ( t ) = y ( t ) + u _ { 0 }$

Also, ${ \dot { y } } ( t ) = { \dot { u } } ( t ) = \Phi ( u ( t ) ) = \Phi ( y ( t ) + u _ { 0 } ) = : { \pmb f } ( t , y , u _ { 0 } )$ . The unique solvability of (2.7) and the continuous dependence of solutions on the initial data $u _ { 0 } \in K$ (see [149]) imply that $u ( t ) \in$ $S ( t ) ( K ) \subset \mathbb { R } ^ { n }$ for every $t \in J$ . The assumption that $\Phi$ be real-analytic on compact subsets of $\mathbb { R } ^ { n }$ implies, via the usual covering and power-series argument, that Φ admits a unique complex-analytic continuation (again denoted by Φ) to a open, complex neighborhood $\begin{array} { r } { K _ { \mathbb { C } } \subset \mathbb { C } ^ { n } \operatorname { o f } \bigcup _ { 0 < t < T } S ( t ) ( K ) \subset } \end{array}$ $\mathbb { R } ^ { n }$ . It also implies that for every $u _ { 0 } \in K$ , the solution $t \mapsto u ( t )$ is analytic w.r. to $\bar { t \in J . }$

This gives that $\pmb { f } ( t , y , u _ { 0 } ) = \pmb { \Phi } ( y ( t ) + u _ { 0 } )$ is complex-analytic w.r. to $( t , y , u _ { 0 } )$ . Now [149, Chap.III.13, Thm. XI] yields that the solution $y ( t )$ depends holomorphically on $\lambda \in K$ . Due to $u ( t ) = y ( t ) + u _ { 0 } = y ( t ) + \lambda$ , we infer that for $u _ { 0 } ~ \in ~ K _ { \mathbb { C } }$ , the solution $u ( t ) ~ = ~ S ( t ) u _ { 0 }$ depends holomorphically on $u _ { 0 }$ . Upon restricting to $u _ { 0 } \in K \subset \mathbb { R } ^ { n }$ , we obtain the claimed analytic dependence of $u ( t ) = S ( t ) u _ { 0 }$ on $u _ { 0 } \in K$ □

The initial-data to solution map $S ( t )$ being an analytic function of (the components of) $u _ { 0 } \in K$ allows us to deduce statements on approximation rates of $S ( t )$ by finite-parametric surrogates, in particular of polynomial and neural network type.

To state a concrete result, we assume

$$
K = [ - 1 , 1 ] ^ { n }
$$

. It is classical that functions which are real-analytic for $t \in [ - 1 , 1 ]$ can be approximated at an exponential rate by polynomials of degree $p \geq 1$ in the uniform norm on $[ - 1 , 1 ] \ ( \mathrm { s e e } , \mathrm { e . g . } , [ 3 5 , 1 2 . 4 . 7 ] )$ It follows by tensorization that, for $t > 0$ and $u _ { 0 } \in K = [ - 1 , 1 ] ^ { n }$ , the component $j \in \{ 1 , \ldots , n \}$ of the map $S ( t ) : K  \mathbb { R } ^ { n }$ can be approximated in the uniform norm on K by a polynomial $S _ { k j } ( t ) ( u _ { 0 } )$ of degree k (interpolating, e.g. in a tensor-product grid of Chebyšev - nodes of order k in K) with respect to the “input” $u _ { 0 } \in K = [ - 1 , 1 ] ^ { n }$ according to

$$
\operatorname* { s u p } _ { u _ { 0 } \in K } | ( S ( t ) ( u _ { 0 } ) ) _ { j } - S _ { k j } ( t ) ( u _ { 0 } ) | \leq C \exp ( - b k )
$$

where the constants b, $C > 0$ depend on $t > 0 , L _ { \Phi }$ and the modulus of the analytic continuation of $S ( t ) ( \cdot )$ to $K _ { \mathbb { C } } \subset \mathbb { C } ^ { n }$ . The maps $S _ { k j } ( t ) ( \cdot )$ are tensor product polynomials of degree $k \geq 1$ in $K$ hence require $O ( k ^ { n } )$ many parameters.

Approximating in this way each component $( S ( t ) ( \cdot ) ) _ { j = 1 } ^ { n }$ of the data-to-solution map $S ( t ) ( \cdot )$ yields a spectral surrogate of the solution map of the ODE-IVP (2.7) with altogether $O ( k ^ { n + 1 } )$ degrees of freedom.

Theorem 2.5 (Spectral Surrogate). For the autonomous ODE-IVP (2.7) with globally Lipschitz Φ which is, in addition, componentwise real analytic on compact subsets $o f \mathbb { R } ^ { n }$ , for arbitrary, fixed $t > 0$ , the solution map $S ( t ) : u _ { 0 } \mapsto u ( t )$ on the set of initial data $K = [ - 1 , 1 ] ^ { n }$ can be approximated by a finite-parametric, spectral surrogate map $S _ { k } ( t )$ which is, component-wise, a tensor product polynomial of degree $k \geq 1$ , with worst-case error bound

$$
\operatorname* { s u p } _ { u _ { 0 } \in K } \| S ( t ) ( u _ { 0 } ) - S _ { k } ( t ) ( u _ { 0 } ) \| _ { \infty } \leq C \exp ( - b k ) ,\tag{2.10}
$$

with constants $b , C > 0$ independent of k, but depending (exponentially, in general) on $t , L _ { \Phi }$

In terms of the number of parameters N of $S _ { k }$ , it holds $N ( k ) = O ( n k ^ { n } )$ . This implies in (2.10) that there exist constants $b ^ { \prime } , C > 0$ such that

$$
\operatorname* { s u p } _ { u _ { 0 } \in K } \| S ( t ) ( u _ { 0 } ) - S _ { k } ( t ) ( u _ { 0 } ) \| _ { \infty } \leq C \exp ( - b ^ { \prime } N ^ { 1 / ( n + 1 ) } ) , \quad 0 \leq t \leq T .\tag{2.11}
$$

Here, $b ^ { \prime } = b \exp ( - ( n + 1 ) \log ( n + 1 ) )$ tends to zero as $n \to \infty$ at least exponentially.

Remark 2.6. (i) Curse of Dimensionality: The shrinkage of the constant b<sup>′</sup> in (2.11) as $n \to \infty$ and the decrease in the exponential rate $\exp ( - b ^ { \prime } N ^ { 1 / ( n + 1 ) } )$ as $n \to \infty$ is a form of the CoD. It precludes direct transfer of Thm. $\it 2 . 5$ to infinite-dimensional sets of initial data and state spaces. A large part of construction of operator surrogates in SciML revolves around the handling $n \to \infty$ without the CoD.

(ii) Constructive Surrogate: The proof of existence of the surrogate map $S _ { k }$ is constructive: it requires solving (numerically, in parallel) $( k + 1 ) ^ { n }$ many instances of the $O D E \ – I V P \ ( 2 . 7 )$ in the unisolvent, for the tensor-product polynomials of degree k, (isotropic) tensor-product Chebyšev-grid in $K = [ - 1 , 1 ] ^ { n }$

(iii) Uniform w.r.t. $t \in [ 0 , T ]$ Surrogate: So $f a r ,$ we considered $\{ S _ { k } ( t ) \} _ { k \geq 1 }$ for $t \in ( 0 , T ]$ fixed. In Theorem ${ \it 2 . 5 , }$ the dependence of $u ( t )$ on $t \in [ 0 , T ]$ is analytic. This may be leveraged to construct exponentially consistent solution operator surrogates $S _ { k } : = \{ S _ { k } ( t ) : 0 \leq t \leq T \}$ $k = 1 , 2 , . . . , e . g $ . by Chebyšev-interpolation of the $S _ { k } ( t )$ in the Chebyšev-points of order k in $[ 0 , T ]$

(iv) Work Bound: $T h e$ total work for determining, for arbitrary, fixed $t \in ( 0 , T ]$ , the surrogate map $S _ { k } ( t )$ with accuracy $\varepsilon \in ( 0 , 1 ]$ , i.e.

$$
\operatorname* { s u p } _ { u _ { 0 } \in K } \| u ( t ) - S _ { k } ( t ) u _ { 0 } \| _ { \infty } \le \varepsilon ,
$$

is bounded a-priori by $O ( 1 + \log ( \varepsilon ) ^ { b } )$ for some constant $b ( n ) > 0$ . Here, $b ( n ) = O ( n )$ as n increases. This follows upon training with approximate solutions of the $I V P \left( 2 . 7 \right)$ . Accuracy of training data computed, $e . g .$ , by an initial-value ODE solver, in $[ 0 , T ]$ is to be consistent with the target accuracy (2.11). The exponential surrogate-consistency shown $e . g .$ in Theorem 2.7 needs to be ensured for the data used in the “training” process. Generating training trajectories with standard, fixed order ODE solvers such as Runge-Kutta methods, which converge at finite, algebraic order in terms of work vs. accuracy, even for analytic Φ, is possible to this end. However, ensuring possibly high fidelity ε of training data will entail possibly very large, costly “ofline” computations. Once the (small, in terms of NN size vs. error ε) surrogate maps $S _ { k } ( t )$ have been determined the predictive evaluation of $S _ { k } ( t )$ for specific initial data is considerably faster than “simulation from scratch”: this qualitative property of operator surrogates is one principal “raison d’être” of these surrogates. Similar comments apply to MOR approaches (see, e.g., [127] and the references there).

Spectral approaches to nonlinear IVPs such as those developed in [132], will furnish such training data, however, in polylog w.r. to ε work vs. accuracy.

With the spectral surrogates $S _ { k } ( t )$ of the nonlinear initial-data to solution map $S ( t )$ at hand, there arises the question on other ways to approximate $S ( t )$ . For SciML, particular interest is in deep neural network surrogates.

A well-known result, due to Telgarsky and Yarotsky [140, 155] states that ReLU NNs can emulate polynomials of degree k in $[ - 1 , 1 ]$ at exponential rate in the uniform norm in terms of the number of trainable network parameters<sup>1</sup> See e.g. [117] and the references there for details. From this result, with a tensor-product construction (suitably modified, as ReLU NNs do not allow a mathematically exact multiplication of two real numbers), with the triangle inequality and Theorem 2.5 follows a corresponding existence result for ReLU NN surrogate maps. As is common, we refer in the following to the size of the network as the number of its trainable parameters, e.g., [123, Chapter 2].

Theorem 2.7. Under the assumptions of Theorem 2.5, for every $t > 0$ and for every $\varepsilon \in ( 0 , 1 / 2 ]$ there exists a fully connected, feedforward ReLU NN surrogate map $\tilde { S } _ { k } ( t )$ with n input and output channels such that

$$
\| S ( t ) - \tilde { S } _ { k } ( t ) \| _ { W ^ { 1 , \infty } ( [ - 1 , 1 ] ^ { n } , \mathbb { R } ^ { n } ) } \leq C \varepsilon .
$$

The size of the $N N \ \tilde { S } _ { k } ( t )$ measured in terms of the number $N ( \varepsilon )$ of neurons equals $O ( | \log ( \varepsilon ) | ^ { n + 1 } )$ .

Proof. We fix $t > 0$ and recall that the map $S ( t ) : K  \mathbb { R } ^ { n }$ is real analytic, component-wise. It hence admits a holomorphic extension to a neighborhood $K _ { \mathbb { C } }$ of K in $\mathbb { C } ^ { n }$ . Therefore, there exists a poly-ellipse $\mathcal { E } _ { \rho } \subset \mathbb { C } ^ { n }$ such that

$$
K = [ - 1 , 1 ] ^ { n } \subset { \mathcal { E } } _ { \rho } \subset K _ { \mathbb { C } } \subset \mathbb { C } ^ { n }
$$

$$
\rho > 1
$$

$$
S ( t )
$$

$$
\overline { { \mathcal { E } _ { \rho } } } ,
$$

Applying [118, Theorem 3.6] to emulate each component of $S ( t )$ , we obtain that

$$
\begin{array} { r } { \| S ( t ) - \tilde { S } _ { k } ( t ) \| _ { W ^ { 1 , \infty } ( [ - 1 , 1 ] ^ { n } , \mathbb { R } ^ { n } ) } \leq C \exp ( - b N ^ { 1 / ( n + 1 ) } ) , } \end{array}
$$

with constants $b = b ( \rho , n ) > 0$ and $C = C ( S , \rho , n )$ . By [118, Theorem 3.6] it further holds that

$$
\mathrm { s i z e } ( \tilde { S } _ { k } ( t ) ) \le N , \quad \mathrm { d e p t h } ( \tilde { S } _ { k } ( t ) ) \le C N ^ { \frac { 1 } { n + 1 } } \log ( N ) ,
$$

where the size refers again to the number of trainable parameters, and depth refers to the depth of the neural network $\tilde { S } _ { k }$ . We then obtain the result by choosing N such that $C \exp ( - b N ^ { 1 / ( n + 1 ) } ) \leq \varepsilon$ □

The preceding example featured key ideas in the construction of neural operators: a mathematically well-defined map $( S ( t )$ in (2.8)) is replaced by a hierarchical sequence of finite-parametric, computable surrogate maps $( S _ { k } ( t )$ in Theorem 2.5) with convergence guarantees (the error bounds (2.10), (2.11)). They also indicate several so-called approximation architectures (spectral- and neural) for the construction of finite-parametric surrogates. The surrogates $S _ { k }$ are constructed for a fixed time-horizon T, but embody in principle full knowledge on the evolution of solutions, for all times $t > 0$ , on ground of a) their convergence, as $k  \infty$ , to the exact evolution map, and b) the semigroup property (2.9) which could be mimicked in order to obtain evolution maps for arbitrary multiples of T by composition of $S _ { k }$ . Naturally, the error of such compositions may accumulate, and finding stable frameworks for long-time simulation remains an active research area. In this connection, it is important that both, polynomials and deep neural networks, are closed under composition, if we allow increases in polynomial degree and neural network depth, respectively.

In the preceding example, both inputs and output were from possibly high-, but finite-dimensional Euclidean space. Once partial diferential equations (PDEs) are considered, infinite-dimensional inand outputs must be admitted, as input and output spaces of PDEs are function spaces. This mandates extra building blocks in the construction of surrogates: on the input side, encoders which sample function space input data, and reduce it to finite-parametric signals (typically, finite sequences of expansion coeficients, or “signals”) which, subsequently, are processed by a parametric surrogate map of the type of $S _ { k }$ considered above. Finally, each output vector provided by $S _ { k }$ must, in turn, be decoded into a corresponding element of the output (function) space, by a decoder block. This naturally leads to the so-called encoder-approximator-decoder architectures, which are shared by a large number of neural operator which have been proposed recently. We conclude discussion of this simple example with the remark, that the ODE initial-data-to-solution surrogate can be used as a building block in operator networks for the solution of linear transport equations with non-constant advection field as considered in [83]. For analytic transport fields, Theorem 2.7 again implies exponential expressivity. Alternative operator network constructions for linear, hyperbolic evolution PDEs leverage NN approximations in mathematical solution representations, e.g. in linear PDEs via fundamental solutions or superpositions of waves. See, e.g., [25, 24] and the references there.

\- As an illustrative example of an IVP we consider the Lotka-Volterra equations in notebook 6\_IVP.ipynb. These equations model the dynamics of a population of two species in a predatorprey relationship. The notebook further provides experiments in which the mapping of an initial state to the state at a time T is approximated with a polynomial interpolant and with a feedforward network.

## 2.3 Parabolic Initial-Boundary Value Problems

Operator Learning within the present, abstract framework is also possible for evolution PDEs. Consider the nonstationary version of the difusion equation (2.2). With the notation of Section 2.1, we are additionally given a finite time-horizon $0 < T < \infty$ and a space-time cylinder $( 0 , T ) \times \mathrm { D }$ We assume given a (time-independent, for simplicity) difusion coeficient $a \in L _ { a _ { - } } ^ { \infty } ( \mathrm { D } )$ as in (2.1), initial data $y _ { 0 } \in H : = L ^ { 2 } ( \mathrm { D } )$ , and a time-dependent source $f ( \eta , t )$ with $f \in L ^ { 2 } ( 0 , T ; H )$

Then the linear, parabolic initial-boundary value problem reads: find $y ( \eta , t )$ such that

$$
\begin{array} { r l r l } { \partial _ { t } y - \nabla _ { \eta } \cdot ( a ( \eta ) \nabla _ { \eta } y ) = f } & { } & & { \mathrm { i n ~ } ( 0 , T ) \times \mathrm { D } , } \\ { y ( \cdot , t ) | _ { \partial \mathrm { D } } = 0 } & { } & & { \mathrm { f o r ~ } 0 < t \leq T , } \\ { y ( \cdot , 0 ) = y _ { 0 } } & { } & & { \mathrm { i n ~ t h e ~ s e n s e ~ o f } ~ H \ . } \end{array}\tag{2.12}
$$

For fixed, suficiently regular source term f and for fixed $y _ { 0 } \in H$ , the evolution equation (2.12) is well-posed, for $a \in L _ { a _ { - } } ^ { \infty } ( \operatorname { D } ) \subset \mathcal { X } = L ^ { \infty } ( \operatorname { D } )$ , with the solution space $\mathcal { V }$ now given by the intersectiontype Bochner space

$$
\mathcal { V } = L ^ { 2 } ( 0 , T ; H _ { 0 } ^ { 1 } ( \mathrm { D } ) ) \cap H ^ { 1 } ( 0 , T ; H ^ { - 1 } ( \mathrm { D } ) ) .\tag{2.13}
$$

We refer to [134, Sec. 5] and the references there for the proof of unique solvability and to [64, Sec. 2.3] for a proof of analyticity of the coeficient-to-solution map $\mathcal { G } : L _ { a _ { - } } ^ { \infty } ( \mathrm { D } ) \to \mathcal { Y } : a \mapsto y .$

Corresponding space-time formulations for nonlinear evolution equations are found in [135].

## 3 Representation Systems

In practice, any computable<sup>2</sup> approximation $\hat { \mathcal G }$ of $\mathcal { G } : \mathcal { X }  \mathcal { Y }$ requires discretization, as a digital computer cannot directly handle infinite dimensional objects, such as elements in function spaces. The first discretization concerns the operator input in  and performs a dimension reduction. Input discretization is referred to as the encoder, . The encoder is a mapping from

$$
\mathcal { E } : \mathcal { X }  \mathbb { R } ^ { d _ { \mathcal { X } } } ,
$$

for some $d _ { \mathcal { X } } \in \mathbb { N }$ . The encoded input is subsequently transformed by an approximator or coeficient map

$$
g : \mathbb { R } ^ { d _ { \mathcal { X } } }  \mathbb { R } ^ { d _ { \mathcal { Y } } } ,
$$

for some $d _ { \mathcal { Y } } \in \mathbb { N }$ . To obtain an element of the output space , we finally apply a decoder

$$
\mathcal { D } : \mathbb { R } ^ { d y }  \mathcal { V } .
$$

Any operator approximation scheme which we consider here is then of the generic “Encoder-Approximator-Decoder” form

$$
\hat { \mathcal { G } } = \mathcal { D } \circ g \circ \mathcal { E } \simeq \mathcal { G } .\tag{3.1}
$$

For particular architectures, the choice of encoder and decoder plays a central role, in particular with respect to how the inputs (from function space) are accessed (via the encoder ), and how the outputs (into function space) are represented (via the decoder ). For other architectures, and are only given implicitly and may not even be explicitly mentioned in the description of the architecture. Moreover, the precise boundary between what is considered part of the encoder or decoder and what is part of g is generally not canonical and to some extent subject to interpretation.

If and are function spaces, the encoder typically evaluates a function at a finite set of sampling points, possibly followed by some form of transformation such as, e.g., a linear or nonlinear dimension reduction. The decoder usually either maps a finite set of function values to some form of interpolant (e.g., a piecewise multilinear interpolant), or takes coeficients in a chosen representation system, and maps them via a linear or nonlinear transformation to an element of . The encoder and decoder may be fixed a priori or learned from data. The coeficient map g is typically learned. In this and the following section we discuss some concrete examples.

## 3.1 Stable linear representations

A popular class of encoder/decoder maps in scientific machine learning is made up by bounded linear maps. In this case

$$
\begin{array} { r } { \mathcal { E } : = \left\{ \begin{array} { l l } { \mathcal { X } \to \mathbb { R } ^ { d _ { \mathcal { X } } } } \\ { a \mapsto ( \varphi _ { j } ( a ) ) _ { j = 1 } ^ { d _ { \mathcal { X } } } } \end{array} \right. \qquad \mathrm { ~ a n d ~ } \qquad \mathcal { D } : = \left\{ \begin{array} { l l } { \mathbb { R } ^ { d _ { \mathcal { Y } } } \to \mathcal { Y } } \\ { ( c _ { j } ) _ { j = 1 } ^ { d _ { \mathcal { Y } } } \mapsto \sum _ { j = 1 } ^ { d _ { \mathcal { y } } } c _ { j } \psi _ { j } , } \end{array} \right. } \end{array}
$$

with $\varphi _ { j } \in { \mathcal { X } } ^ { \prime }$ and $\psi _ { j } \in \mathcal { V }$ for all $j .$

Encoder and decoder sequences must be chosen carefully to capture the essential features of the operator’s input and output: any information in the kernel of is lost, and anything outside the range of $\mathcal { D }$ cannot be approximated by (3.1). It is thus clear that (3.1) can only provide a good approximation if ker( ) spans a subspace of input features that are of little importance, while im( ) describes the most important features of the output. We therefore wish to understand which functionals $\varphi _ { j }$ provide the most information to reconstruct a from $( \varphi _ { j } ( a ) ) _ { j = 1 } ^ { m }$ . To make this mathematically concrete, we introduce the concept of Banach frames. We opt for the definition in [60, Section 1], and assume the Banach spaces  and  to be isomorphic to (weighted) ℓ<sup>p</sup> sequence space. This includes a wide range of separable function spaces of Sobolev- and Besov-Triebel-Lizorkin type, see [143, 144] and the references there.

Definition 3.1. Let be a separable Banach space<sup>3</sup> and let $p \in [ 1 , \infty )$

A family $( \varphi _ { \mathcal { Z } , j } ) _ { j \in \mathbb { N } }$ of functionals in $\mathcal { Z } ^ { \prime }$ and elements $( \psi _ { \mathcal { Z } , j } ) _ { j \in \mathbb { N } }$ in $\mathcal { Z }$ is called a p-Banach frame for , if

(i) the analysis operator

$$
A _ { \mathcal { Z } } : \mathcal { Z } \to \ell ^ { p } ( \mathbb { N } ) : z \mapsto ( \varphi z , j ( z ) ) _ { j \in \mathbb { N } }
$$

is continuous,

(ii) the norms $\| z \| _ { \mathcal { Z } }$ and $\| ( \varphi z , j ( z ) ) _ { j \in \mathbb { N } } \| _ { \ell ^ { p } }$ are equivalent on ${ \mathcal { Z } } ,$

(iii) the synthesis operator

$$
S _ { \mathcal { Z } } : \ell ^ { p } ( \mathbb { N } )  \mathcal { Z } : ( c _ { j } ) _ { j \in \mathbb { N } } \mapsto \sum _ { j \in \mathbb { N } } c _ { j } \psi _ { \mathcal { Z } , j }
$$

is a bounded linear map satisfying

$$
S z ( A z ( z ) ) = z\tag{3.2}
$$

for all $z \in { \mathcal { Z } }$

Linear Encoders $\mathcal { E } _ { \mathcal { Z } }$ can be obtained by combining $A _ { \mathcal { Z } }$ with $d _ { \mathcal { Z } } .$ term truncation to the first $d _ { \mathcal { Z } }$ coeficients of the output of $A _ { \mathcal { Z } }$ . Linear Decoders $\mathcal { D } _ { \mathcal { Z } }$ can be obtained by padding a finite sequence in $\mathbb { R } ^ { d _ { Z } }$ with zeros to an element of $\ell ^ { p } ( \mathbb { N } )$ , and then applying $S z$

Remark 3.2. If $p = 2$ and $\mathcal { Z }$ is a separable Hilbert space, then $( i )$ and $( i i )$ imply the existence of $( \psi _ { \mathcal { Z } , j } ) _ { j \in \mathbb { N } }$ satisfying (iii), see e.g. [66].

For a p-Banach frame $( \varphi _ { \mathcal { X } , j } , \psi _ { \mathcal { X } , j } ) _ { j \in \mathbb { N } }$ on $\mathcal { X } ,$ one can estimate the information lost when working with the d-term truncations of (sequences of) coeficients $( \varphi _ { \mathcal { X } , j } ( a ) ) _ { j = 1 } ^ { d }$ as follows: Motivated by (3.2), we define a reconstruction map R via

$$
R : = \left\{ \begin{array} { l l } { \mathbb { R } ^ { d } \to \mathcal { Z } } \\ { ( c _ { j } ) _ { j = 1 } ^ { d } \mapsto \sum _ { j = 1 } ^ { d } c _ { j } \psi _ { \mathcal { Z } , j } . } \end{array} \right.
$$

Then Definition 3.1 (iii) implies the existence of a constant C such that for all $a \in { \mathcal { X } }$

$$
\begin{array} { r l } & { \| a - R ( \varphi x , j ( a ) ) \| _ { \mathcal { X } } = \| S _ { \mathcal { X } } ( A _ { \mathcal { X } } ( a ) ) - S _ { \mathcal { X } } ( ( \varphi x , 1 ( a ) , \dotsc , \varphi x , d ( a ) , 0 , 0 , \dotsc ) \| _ { \mathcal { X } } } \\ & { \qquad \leq C \| ( \varphi x , j ( a ) ) _ { j > d } \| _ { \mathcal { P } } . } \end{array}
$$

This provides a rough, first guiding criterion for determining how many coeficients d are needed<sup>4</sup> to retain suficient information about the operator input $a \in { \mathcal { X } }$ . Analogous considerations hold for the representation of the operator output y via a p-Banach frame $( \varphi y , j , \psi y , j ) _ { j \in \mathbb { N } }$ in $\mathcal { V }$ . We will discuss this in more detail in Section 4. This motivates the en- and decoder choice

$$
\mathcal { E } ( a ) = ( \varphi x , j ( a ) ) _ { j = 1 } ^ { d x } \qquad \mathrm { a n d } \qquad \mathcal { D } ( ( c _ { j } ) _ { j = 1 } ^ { d y } ) = \sum _ { j = 1 } ^ { d y } c _ { j } \psi _ { \mathcal { Y } , j } .\tag{3.3}
$$

The appeal of using Banach frames in operator learning lies in the continuity of both the coeficientto-function map and its inverse, which ensures stability in either direction.

Example 3.3. Typical examples include the following:

• Orthonormal bases in Hilbert spaces: Suppose that $( \psi _ { j } ) _ { j \in \mathbb { N } }$ is an orthonormal basis of the separable Hilbert space . Using Parseval’s identity, it is straightforward to check that

$$
\varphi _ { j } ( a ) : = \langle a , \psi _ { j } \rangle _ { \mathcal { X } } \qquad \forall j \in \mathbb { N }
$$

forms a 2-Banach frame for  .

• Frames in Hilbert spaces: More generally, let $( \psi _ { j } ) _ { j \in \mathbb { N } }$ be a frame for a Hilbert space X $I 6 6 J ,$ meaning there exist frame bounds $\lambda , \Lambda > 0$ such that

$$
\lambda \| a \| _ { \mathcal { X } } ^ { 2 } \leq \sum _ { j \in \mathbb { N } } | \langle a , \psi _ { j } \rangle _ { \mathcal { X } } | ^ { 2 } \leq \Lambda \| a \| _ { \mathcal { X } } ^ { 2 } \qquad \forall a \in \mathcal { X } .\tag{3.4}
$$

Setting $\varphi _ { j } ( a ) : = \langle a , \psi _ { j } \rangle _ { \mathcal { X } }$ , it is again easy to see that (3.4) guarantees the conditions of $D e f i -$ nition 3.1 for $p = 2$

![](images/8267321159a13fac918f86d77dbb71cc8133ff34d2a15d41aaabfe17616b5474.jpg)  
Figure 1: Continuous, piecewise linear wavelet basis of $\mathcal { X } = H _ { \{ 0 \} } ^ { 1 } ( 0 , 1 )$ on subdivision level $j = 2$ in $I = ( 0 , 1 )$ with two vanishing moments and a homogeneous Dirichlet boundary condition in $t = 0$ (imposed only on the left boundary wavelet). See the appendix of [65] for details and [146, 33, 113] for other, spline- and Finite-Element like constructions.

• Besov-Triebel-Lizorkin spaces: Let $\chi = B _ { p , q } ^ { s } ( \mathrm { D } )$ (Besov space) or $\chi = F _ { p , q } ^ { s } ( \mathrm { D } )$ (Triebel-Lizorkin space) for suitable parameters $( s , p , q )$ . For such spaces there exists (on suitable domains $D ,$ see [143, 144] for details) a countable system of functions $( \psi _ { j } ) _ { j \in \mathbb { N } }$ (e.g., wavelets) such that any $a \in { \mathcal { X } }$ can be represented as

$$
a = \sum _ { j \in \mathbb { N } } c _ { j } \psi _ { j } ,
$$

and the norm $\| a \| _ { \mathcal { X } }$ is equivalent to a (possibly weighted) ℓ<sup>p</sup>-norm of the coeficient sequence $( c _ { j } ) _ { j \in \mathbb { N } }$ . This isomorphism between the function space and a sequence space provides the structure of a Banach frame. See $I 1 \llangle 3 ,$ , Corollary ${ 5 . 4 4 } { } J$ for more details.

• Biorthogonal spline-wavelet bases of hilbertian Sobolev spaces in domains: These systems are Riesz-bases and, formally, special cases of frames. Concrete constructions of such systems have been developed over the past decade, we refer to [130, 33] and references there for examples. Figure 1 shows a simple biorthogonal spline wavelet system in the univariate domain $D = ( 0 , 1 )$ , which constitutes a Riesz basis for the space $\mathcal { X } = \{ v \in H ^ { 1 } ( \mathrm { D } ) \mid v ( 0 ) = 0 \}$ We refer to [134, Sec. 6] for a construction of space-time spline-wavelet systems for the Bochner space in (2.13) for parabolic initial-boundary value problems such as (2.12).

\- We set up an encoder $\mathcal { E }$ for the parametric difusion problem based on a priori assumption on the input distribution in notebook 1\_Representation.ipynb. Specifically, we assume that the log-normal difusion coeficient log a can be expressed in the orthonormal sine basis $\psi _ { j } ( x ) = $ ${ \sqrt { 2 } } \sin ( j \pi x )$ of $L ^ { 2 } ( 0 , 1 )$ , i.e., log $\begin{array} { r } { a ( x ; y ) = \sum _ { j = 1 } ^ { d } y _ { j } \psi _ { j } ( x ) } \end{array}$ , where $y = ( y _ { 1 } , \dots , y _ { d } ) \in \mathbb { R } ^ { d }$ is the parameter vector. The encoder then maps a difusion coeficient $a$ to $\mathrm { ( a }$ truncated version of) the parameter vector. The log-normal form guarantees $a > 0$ pointwise, as required by (2.1), and the basis $( \psi _ { j } ) _ { j \geq 1 }$ is an instance of the orthonormal basis frame in Example 3.3.

## 3.2 PCA

The approach in Section 3.1 is sensible when Banach frames of $\mathcal { X }$ and $\mathcal { V }$ are known a priori and constructable. However, in many applications the inputs of $\mathcal { G }$ are assumed to be i.i.d. realizations of a random variable taking values in $\mathcal { X }$ with i.i.d. understood with respect to a probability measure $\mu$ on $\mathcal { X } .$ . The support of a finite sample of input data for $\mathcal { G }$ may occupy only a small subset of $\mathcal { X }$ . Similarly, the outputs follow the pushforward measure<sup>5</sup> $\mathcal { G } _ { \sharp } \mu$ , which, depending on ${ \mathcal { G } } ,$ may be supported on a small subset of $\mathcal { V } .$ . In these cases, even if Banach frames of $\mathcal { X }$ and $\mathcal { V }$ are available, they may not yield the most eficient representation systems.

## 3.2.1 Formulation

For eficient compression, we aim to determine a best approximating d-dimensional subspace of a data distribution $\mu$ on a space $\mathcal { Z } .$ . Mathematically, this can be formulated as an optimization problem of the type

$$
\operatorname* { i n f } _ { \substack { V \subseteq \mathcal { Z } , \ \mathrm { d i m } ( V ) = d } } \mathbb { E } _ { z \sim \mu } [ \operatorname* { i n f } _ { v \in V } \Vert z - ( z _ { 0 } + v ) \Vert _ { \mathcal { Z } } ^ { p } ] ,\tag{3.5}
$$

for a suitable $p \geq 1$ (and assuming finite pth moment of $\mu )$ . We are thus looking for the best approximating d-dimensional afine subspace $z _ { 0 } + V$ of $\mathcal { Z } .$ . In general, this optimization problem need not have a unique solution, and even if it does, it might be dificult to find in practice. In the following we thus only focus on the case in which $p = 2$ and $\mathcal { Z }$ is a Hilbert space; This is known as functional PCA, and in this case the solution can be described explicitly. We comment on the usual PCA in finite dimensional spaces in Remark 3.5.

To explain how, we assume $\mu$ is a probability measure on $\mathcal { Z }$ with finite first and second moments, i.e.

$$
\int _ { \mathcal { Z } } \| z \| _ { \mathcal { Z } } \mathrm { d } \mu ( z ) < \infty , \qquad \int _ { \mathcal { Z } } \| z \| _ { \mathcal { Z } } ^ { 2 } \mathrm { d } \mu ( z ) < \infty .\tag{3.6}
$$

For a d-dimensional subspace $V \subseteq { \mathcal { Z } }$ and with the orthogonal projection $P _ { V } : { \mathcal { Z } } \to V$ it holds for any $z \in { \mathcal { Z } }$

$$
\| z - P _ { V } z \| _ { \mathcal { Z } } = \operatorname* { i n f } _ { v \in V } \| z - v \| _ { \mathcal { Z } } .
$$

Therefore, the objective in (3.5) corresponds to

$$
\mathbb { E } _ { z \sim \mu } \| z - z _ { 0 } - P _ { V } ( z - z _ { 0 } ) \| _ { \mathcal { Z } } ^ { 2 } = \mathbb { E } _ { z \sim \mu } \| ( z - P _ { V } z ) - ( z _ { 0 } - P _ { V } z _ { 0 } ) \| _ { \mathcal { Z } } ^ { 2 } .
$$

The element in $\mathcal { Z }$ best approximating the random variable $\tilde { z } : = z - P _ { V } z$ in the least-squares sense is $\mathbb { E } [ \tilde { z } ] = \mathbb { E } [ z ] - \mathbb { E } [ P _ { V } z ] = \mathbb { E } [ z ] - P _ { V } \mathbb { E } [ z ]$ (see Lemma 3.9 ahead). For this reason, we fix

$$
z _ { 0 } = \mathbb { E } _ { z \sim \mu } [ z ] = \int _ { \mathcal { Z } } z \mathrm { d } \mu ( z ) .\tag{3.7}
$$

By Pythagoras’ theorem

$$
\lVert z - P _ { V } z \rVert _ { \mathcal { Z } } ^ { 2 } = \lVert z \rVert _ { \mathcal { Z } } ^ { 2 } - \lVert P _ { V } z \rVert _ { \mathcal { Z } } ^ { 2 } .
$$

The optimization problem (3.5) for $p = 2$ can now be simplified to

$$
\operatorname* { i n f } _ { V \subseteq Z , \ \dim ( V ) = d } \mathbb { E } _ { z \sim \mu \left[ \left. ( z - z _ { 0 } ) - P _ { V } ( z - z _ { 0 } ) \right. _ { Z } ^ { 2 } \right] } = \operatorname* { i n f } _ { V \subseteq Z , \ \dim ( V ) = d } \left( \mathbb { E } _ { z \sim \mu \left[ \left. z - z _ { 0 } \right. _ { Z } ^ { 2 } \right] - \mathbb { E } _ { z \sim \mu } \left[ \left. P _ { V } ( z - z _ { 0 } ) \right. _ { Z } ^ { 2 } \right] } \right) .
$$

Thus, V is equivalently characterized as the subspace maximizing

$$
\mathbb { E } _ { z \sim \mu } [ \| P _ { V } ( z - z _ { 0 } ) \| _ { \mathcal { Z } } ^ { 2 } ] .\tag{3.8}
$$

For this reason, PCA is often described as finding the linear subspace $V$ on which the data projection onto V has maximal variance.

## 3.2.2 Characterization of V

Introduce the covariance operator $\mathcal { C } : \mathcal { Z }  \mathcal { Z }$ of $\mu$ via

$$
\langle { \mathcal C } x , y \rangle _ { { \mathcal Z } } = \int _ { { \mathcal Z } } \langle x , z - z _ { 0 } \rangle _ { { \mathcal Z } } \langle y , z - z _ { 0 } \rangle _ { { \mathcal Z } } \mathrm { d } \mu ( z ) \qquad \forall x , y \in { \mathcal Z } .\tag{3.9}
$$

$\mathrm { A s }$ is well-known, under assumption $( 3 . 6 ) , \mathcal { C } : \mathcal { Z }  \mathcal { Z }$ as in (3.9) is a well-defined, compact, and self-adjoint linear operator, see, e.g. [139] and references there.

Proposition 3.4. Let $\mathcal { Z }$ be a Hilbert space and let $\mu$ be a probability measure on $\mathcal { Z }$ with finite first and second moments. Then, for any $d \in \mathbb { N }$ , a d-dimensional subspace $V \subseteq { \mathcal { Z } }$ maximizes (3.8) if and only if V is a span of eigenvectors of  corresponding to the d largest eigenvalues.

Proof. Since is self-adjoint and compact, there exists a null-sequence of monotonically decreasing, non-negative eigenvalues $( \lambda _ { j } ) _ { j \in \mathbb { N } }$ and corresponding orthonormal eigenvectors $( \psi _ { j } ) _ { j \in \mathbb { N } }$ in $\mathcal { Z } ,$ , such that  has the spectral decomposition

$$
\mathcal { C } z = \sum _ { j \in \mathbb { N } } \lambda _ { j } \langle \psi _ { j } , z \rangle _ { \mathcal { Z } } \psi _ { j } \qquad \forall z \in \mathcal { Z } .
$$

Let V be an arbitrary d-dimensional subspace of ${ \mathcal { Z } } ,$ and fix an orthonormal basis $e _ { 1 } , \ldots , e _ { d }$ of V. Then

$$
\| P _ { V } z \| _ { \mathcal { Z } } ^ { 2 } = \sum _ { i = 1 } ^ { d } \langle z , e _ { i } \rangle ^ { 2 } \qquad \forall z \in \mathcal { Z } ,
$$

and thus

$$
\mathbb { E } _ { z \sim \mu } [ \| P _ { V } ( z - z _ { 0 } ) \| _ { Z } ^ { 2 } ] = \int _ { \mathcal { Z } } \sum _ { i = 1 } ^ { d } \langle z - z _ { 0 } , e _ { i } \rangle ^ { 2 } \mathrm { d } \mu ( z ) = \sum _ { i = 1 } ^ { d } \langle \mathcal { C } e _ { i } , e _ { i } \rangle = \sum _ { j \in \mathbb { N } } \lambda _ { j } \underbrace { \sum _ { i = 1 } ^ { d } \langle \psi _ { j } , e _ { i } \rangle ^ { 2 } } _ { = : c _ { j } } .
$$

Due to the orthonormality of the $( \psi _ { j } ) _ { j \in \mathbb { N } }$ and the $( e _ { i } ) _ { i = 1 } ^ { d }$ , we have $0 \leq c _ { j } \leq 1$ for all $j \in \mathbb N$ and $\textstyle \sum _ { j \in \mathbb { N } } c _ { j } \ \leq \ d .$ Assume for the moment that the $\lambda _ { j }$ are strictly monotonically decreasing, i.e. $\lambda _ { 1 } > \lambda _ { 2 } > . . . .$ Then the sum $\textstyle \sum _ { j \in \mathbb { N } } \lambda _ { j } c _ { j }$ is maximized under the above constraints on the $( c _ { j } ) _ { j \in \mathbb { N } }$ if and only if $c _ { 1 } = \cdots = c _ { d } = 1$ and $c _ { j } = 0$ for $j > d .$ This is achieved if and only if $\psi _ { j } \in \operatorname { s p a n } \{ e _ { 1 } , . . . , e _ { d } \}$ for all $j = 1 , \ldots , d .$ In other words $V = \operatorname { s p a n } \{ \psi _ { 1 } , \dots , \psi _ { d } \}$

Finally, if the eigenvalues are not necessarily strictly decreasing, i.e. $\lambda _ { 1 } \geq \lambda _ { 2 } \geq . . .$ , the optimal subspace is not necessarily unique. The characterization remains valid however: any subspace spanned by an orthonormal set of eigenvectors corresponding to the d largest eigenvalues yields a maximizer. □

Remark 3.5. Consider the special case of $\mathcal { Z } = \mathbb { R } ^ { n }$ and $\begin{array} { r } { \mu = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \delta _ { z _ { j } } } \end{array}$ being a discrete probability measure on $\mathbb { R } ^ { n }$ (here $\delta _ { z _ { j } }$ denotes the Dirac delta at $z _ { j } \in \mathbb { R } ^ { n } )$ . In this case $\mathcal { C } : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ is defined by its action on all x, $\boldsymbol { y } \in \mathbb { R } ^ { n }$ via<sup>6</sup>

$$
\langle { \mathcal C } x , y \rangle = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } x ^ { \top } ( z _ { j } - z _ { 0 } ) ( z _ { j } - z _ { 0 } ) ^ { \top } y ,\tag{3.10}
$$

where $\begin{array} { r } { z _ { 0 } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } z _ { j } \in \mathbb { R } ^ { n } } \end{array}$ . Letting $A \in \mathbb { R } ^ { n \times m }$ be the matrix with columns $z _ { j } - z _ { 0 }$ , the covariance operator corresponds to

$$
{ \frac { 1 } { m } } \sum _ { j = 1 } ^ { m } ( z _ { j } - z _ { 0 } ) ( z _ { j } - z _ { 0 } ) ^ { \top } = { \frac { 1 } { m } } A A ^ { \top } \in \mathbb { R } ^ { n \times n } .
$$

The optimal d-dimensional subspace is then obtained by the span of the d leading eigenvectors of (or equivalently of $A A ^ { \top } )$ . For numerical stability, in practice this can be done by computing the singular value decomposition of A.

Remark 3.6 (Error of PCA). Denote again by $\lambda _ { 1 } \geq \lambda _ { 2 } \geq . .$ . the eigenvalues of the covariance operator . As in the preceding proof, it holds that

$$
\mathbb { E } _ { z \sim \mu } [ \| z - z _ { 0 } \| _ { \mathcal { Z } } ^ { 2 } ] = \sum _ { j \in \mathbb { N } } \lambda _ { j } .
$$

Together with the estimates shown in the proof this yields the error

$$
\mathbb { E } _ { z \sim \mu } [ \| z - z _ { 0 } - P _ { V } ( z - z _ { 0 } ) \| _ { \mathcal { Z } } ^ { 2 } ] = \sum _ { j > d } \lambda _ { j }
$$

for an optimal d-dimensional subspace $V .$

## 3.2.3 Computation of Encoder and Decoder

In the case of Hilbert spaces, “optimal” (in the sense of minimizing the least squares objective (3.5) for $p = 2 )$ encoders and decoders can be found via PCA: let $x _ { 0 }$ be the expected value of $\mu$ on $\mathcal { X }$

and let $y _ { 0 }$ the expected value of $\mathcal { G } _ { \sharp } \mu$ on $\mathcal { V } .$ Let the PCA yield the orthonormal vectors $( \psi _ { \mathcal { X } , j } ) _ { j = 1 } ^ { d _ { \mathcal { X } } }$ and $( \psi y , j ) _ { j = 1 } ^ { d y }$ in $\mathcal { X } , \mathcal { y }$ , respectively. Then the corresponding encoder and decoder are given by

$$
\pmb { \mathcal { E } } ( \pmb { x } ) = ( \langle \psi \pmb { x } , j , \pmb { x } - \pmb { x } _ { 0 } \rangle ) _ { j = 1 } ^ { d \pmb { x } } \qquad \mathrm { a n d } \qquad \pmb { \mathcal { D } } ( ( c _ { j } ) _ { j = 1 } ^ { d _ { \pmb { y } } } ) = \pmb { y } _ { 0 } + \sum _ { j = 1 } ^ { d _ { \pmb { y } } } c _ { j } \psi \pmb { y } , j .\tag{3.11}
$$

In practice, $\mu$ and $\mathcal { G } _ { \sharp } \mu$ might be unknown or intractable, but we have access to data samples $( x _ { j } , \mathcal { G } ( x _ { j } ) ) _ { j = 1 } ^ { m }$ with $x _ { j } \overset { \mathrm { i i d } } { \sim } \mu$ . In this case, we can still perform the empirical version of PCA. We outline the procedure for the input data:

(i) Center the data. Let $\bar { x } _ { j } : = x _ { j } - \bar { x }$ , where $\begin{array} { r } { \bar { x } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } x _ { j } } \end{array}$ is the sample average.

(ii) Compute the empirical covariance operator. Let

$$
\hat { \mathcal { C } } x = \frac { 1 } { m - 1 } \sum _ { j = 1 } ^ { m } \left. x , \bar { x } _ { j } \right. _ { \mathcal { X } } \bar { x } _ { j } .
$$

Note that $\hat { \mathcal { C } } : \mathcal { X } \to \operatorname { s p a n } \{ \bar { x } _ { 1 } , \dots , \bar { x } _ { m } \}$ . For numerical computations, $\hat { \mathcal { C } }$ can be represented in a finite dimensional basis of span $\{ \bar { x } _ { 1 } , \ldots , \bar { x } _ { m } \}$ , yielding a matrix.

(iii) Solve the eigenvalue problem. Compute the eigendecomposition of the symmetric matrix $\hat { \mathcal { C } }$ (cf. Remark 3.5)

$$
\hat { \mathcal { C } } \hat { \psi } _ { \mathcal { X } , j } = \hat { \lambda } _ { j } \hat { \psi } _ { \mathcal { X } , j } .
$$

(iv) Select the subspace. Choose the d (orthonormal) eigenvectors $( \hat { \psi } _ { \mathcal { X } , j } ) _ { j = 1 } ^ { d }$ corresponding to the d largest eigenvalues $( \hat { \lambda } _ { j } ) _ { j = 1 } ^ { d }$ . These vectors are used to define the encoder in (3.11).

Remark 3.7. It might seem easier to work with an orthonormal basis than a frame; in particular, frames can be redundant, meaning the expansion of an element of the Hilbert/Banach space in the frame is in general not unique. However, frames ofer the crucial advantage of being computationally tractable. Finding an orthonormal basis of Sobolev spaces on standard domains like cubes is straightforward $( e . g .$ , Fourier modes); however, real-world problems are rarely posed on such simple domains. Finding an orthonormal basis for a Sobolev space on a complicated geometry is a nontrivial numerical computation in itself. Constructions for frames on the other hand are known and computable, based, for example, on triangulations of the domain, e.g., [36, 63]. They thus naturally integrate with existing numerical solvers and representations of numerical PDE solutions.

\- We set up a PCA-based decoder  for the parametric difusion problem in notebook 1\_Representation.ipynb.

## 3.3 Active Subspaces

Recall that our goal is to approximate $\mathcal { G } : \mathcal { X }  \mathcal { Y }$ , with inputs stemming from a measure $\mu$ on $\mathcal { X } .$ To perform a dimension reduction on the input space $\mathcal { X } .$ , PCA determines a best approximating d-dimensional subspace V of with respect to $\mu .$ . This approach is entirely agnostic to , however. Suppose for example that $\mathcal { G }$ is constant along a direction $v \in \mathcal { X }$ , i.e. for every $x \in \mathcal { X }$ the map $t \mapsto \mathcal { G } ( x + t v )$ is constant for $t \in \mathbb { R }$ . Even if v corresponds to an important eigendirection of the covariance of $\mu ,$ it should in this case not be included in the reduced subspace $V .$ , as it provides no information about the output of $\mathcal { G }$

In this section, we present basic results for the active subspace approach, see e.g. [31].

## 3.3.1 Formulation

Assume that $x , y$ are separable Hilbert spaces. To take into account the action of ${ \mathcal { G } } _ { : }$ the active subspace approach replaces the objective (3.5) by

$$
\mathop { \operatorname* { i n f } } _ { \mathop { V \subseteq \mathcal { X } } , \ \dim ( V ) = d } \mathbb { E } _ { x \sim \mu } [ \| \mathcal { G } ( x ) - \bar { \mathcal { G } } _ { V } ( P _ { V } x ) \| _ { \mathcal { V } } ^ { 2 } ] ,\tag{3.12}
$$

where $P _ { V }$ is again the orthogonal projection onto V. In words, we wish to determine a d-dimensional subspace $V \subseteq { \mathcal { X } }$ which is best suited for the approximation of $\mathcal { G }$ under the input measure $\mu .$

Remark 3.8. For the objective (3.12) it sufices to consider linear subspaces V (rather than afine subspaces $z _ { 0 } + V$ as in Section 3.2): for a $s h i f t ~ z _ { 0 }$ , the expression $\bar { \mathcal { G } } ( z _ { 0 } + P _ { V } ( z - z _ { 0 } ) )$ corresponds to some $\tilde { \mathcal { G } } ( P _ { V } z )$ . The shift is thus absorbed by the minimization over $\mathcal { G } _ { V }$ in (3.12).

The next lemma will allow to explicitly construct the optimal $\bar { \mathcal { G } } _ { V }$ in (3.12). The actual construction of $\bar { \mathcal { G } } _ { V }$ is developed following its proof.

Lemma 3.9. Let  be a separable Hilbert space, let $\gamma$ be a probability measure on a measure space $\mathcal { Z }$ , and let $\mathcal { F } : \mathcal { Z }  \mathcal { V }$ be measurable.

Then

$$
\operatorname { a r g m i n } _ { y \in \mathcal { V } } \| \mathcal { F } ( z ) - y \| _ { L ^ { 2 } ( \mathcal { Z } , \gamma ) } = \int _ { \mathcal { Z } } \mathcal { F } ( z ) \mathrm { d } \gamma ( z ) .\tag{3.13}
$$

Proof. Denote the right-hand side in (3.13) by $y ,$ and fix an arbitrary $v \in \mathcal { V }$ with $v \neq 0$ . Then

$$
\int _ { \mathcal { Z } } \Vert \mathcal { F } ( z ) - ( y + v ) \Vert _ { \mathcal { Y } } ^ { 2 } \mathrm { d } \gamma ( z ) = \int _ { \mathcal { Z } } \Vert \mathcal { F } ( z ) - y \Vert _ { \mathcal { Y } } ^ { 2 } - 2 \left. \mathcal { F } ( z ) - y , v \right. + \Vert v \Vert _ { \mathcal { Y } } ^ { 2 } \mathrm { d } \gamma ( z ) .\tag{3.14}
$$

Since $\begin{array} { r } { \int _ { \mathcal Z } \left. \mathcal { F } ( z ) - y , v \right. \mathrm { d } \gamma ( z ) = 0 } \end{array}$ by definition of $y ,$ (3.14) is strictly larger than $\begin{array} { r } { \int _ { \mathcal Z } \| \mathcal F ( z ) - y \| _ { \mathcal { V } } ^ { 2 } \mathrm { d } \gamma ( z ) } \end{array}$

Fix a subspace $V \subseteq { \mathcal { X } }$ of finite dimension $d ,$ and denote by $V ^ { \perp }$ its orthogonal complement. In the following, we let

$$
x \sim \mu , \qquad v = P _ { V } x , \qquad w = P _ { V ^ { \perp } } x ,
$$

so that in particular $x = v + w$ . We denote by $\mu _ { V } : = ( P _ { V } ) _ { \sharp } \mu$ the marginal of $\mu$ on $V$ and by $\mu ^ { v }$ the distribution of $w$ conditioned on $v ;$ this is (for almost every v) well-defined, see, e.g., [14, Vol. II, Section 10.4]. With this convention, the objective in (3.12) can be written as

$$
\int _ { V } \int _ { V ^ { \perp } } \| \mathcal { G } ( v + w ) - \bar { \mathcal { G } } _ { V } ( v ) \| _ { \mathcal { V } } ^ { 2 } \mathrm { d } \mu ^ { v } ( w ) \mathrm { d } \mu _ { V } ( v ) .\tag{3.15}
$$

By Lemma 3.9, the mapping ${ \bar { \mathcal { G } } } _ { V } : V \to \mathcal { V }$ minimizing this objective can be identified as the conditional mean of $\mathcal { G }$ along $v + V ^ { \perp }$ , i.e.

$$
\bar { \mathcal { G } } _ { V } ( v ) = \int _ { V ^ { \perp } } \mathcal { G } ( v + w ) \mathrm { d } \mu ^ { v } ( w ) .\tag{3.16}
$$

With this observation, (3.12) reduces to a minimization only over the d-dimensional subspace $V \subseteq$ . The objective (3.15) can further be rewritten to

$$
\begin{array} { l } { \displaystyle \int _ { V } \int _ { V ^ { \perp } } \| \mathcal { G } ( v + w ) - \mathbb { E } _ { \mu } [ \mathcal { G } ] - ( \bar { \mathcal { G } } _ { V } ( v ) - \mathbb { E } _ { \mu } [ \mathcal { G } ] ) \| _ { \mathcal { V } } ^ { 2 } \mathrm { d } \mu ^ { v } ( w ) \mathrm { d } \mu _ { V } ( v ) } \\ { \displaystyle \quad \quad = \int _ { V } \int _ { V ^ { \perp } } \| \mathcal { G } ( v + w ) - \mathbb { E } _ { \mu } [ \mathcal { G } ] \| _ { \mathcal { V } } ^ { 2 } - 2 \left. \mathcal { G } ( v + w ) - \mathbb { E } _ { \mu } [ \mathcal { G } ] , \bar { \mathcal { G } } _ { V } ( v ) - \mathbb { E } _ { \mu } [ \mathcal { G } ] \right. } \\ { \displaystyle \quad \quad + \| \bar { \mathcal { G } } _ { V } ( v ) - \mathbb { E } _ { \mu } [ \mathcal { G } ] \| _ { \mathcal { V } } ^ { 2 } \mathrm { d } \mu ^ { v } ( w ) \mathrm { d } \mu _ { V } ( v ) } \\ { \displaystyle \quad = \int _ { \mathcal { X } } \| \mathcal { G } ( x ) - \mathbb { E } _ { \mu } [ \mathcal { G } ] \| _ { \mathcal { V } } ^ { 2 } \mathrm { d } \mu ( x ) - \int _ { V } \| \bar { \mathcal { G } } _ { V } ( v ) - \mathbb { E } _ { \mu } [ \mathcal { G } ] \| _ { \mathcal { V } } ^ { 2 } \mathrm { d } \mu _ { V } ( v ) , } \end{array}
$$

where the second equality follows from the definition of $\bar { \mathcal { G } } _ { V }$ in (3.16). Since $\mathbb { E } _ { \mu } [ \mathcal { G } ] = \mathbb { E } _ { \mu _ { V } } [ \bar { \mathcal { G } } _ { V } ]$ , the last integral corresponds to the variance of $\bar { \mathcal { G } } _ { V } ( v )$ with respect to $v \sim \mu _ { V }$ . Thus, the optimization problem (3.12) is equivalent to

$$
\operatorname* { s u p } _ { V \subseteq \mathcal { X } , \ \dim ( V ) = d } \int _ { V } \| \bar { \mathcal { G } } _ { V } ( v ) - \mathbb { E } [ \bar { \mathcal { G } } _ { V } ] \| _ { \mathcal { V } } ^ { 2 } \mathrm { d } \mu _ { V } ( v ) .\tag{3.17}
$$

In words, a solution to this problem is the d-dimensional subspace V along which the variance of <sup>the</sup> <sup>conditional</sup> <sup>expectation</sup> <sup>of</sup> G<sup>,</sup> <sup>obtained</sup> <sup>by</sup> <sup>averaging</sup> <sup>over</sup> $V ^ { \perp }$ , is maximized.

## 3.3.2 Characterization of V

Unfortunately, due to the presence of the possibly nonlinear map ${ \mathcal { G } } _ { : }$ , (3.17) is usually a very nonconvex and hard to solve optimization problem. The idea is to instead maximize an upper bound of (3.17).

Denote in the following

$$
\begin{array} { r l } & { C ^ { 1 } \cap H ^ { 1 } ( \mathcal { X } , \mu ; \mathcal { Y } ) : = \biggr \{ \mathcal { G } : \mathcal { X } \to \mathcal { Y } : D \mathcal { G } \in C ^ { 0 } ( \mathcal { X } , H S ( \mathcal { X } , \mathcal { Y } ) ) , } \\ & { \qquad \quad \displaystyle \int _ { \mathcal { X } } \| \mathcal { G } ( x ) \| _ { \mathcal { Y } } ^ { 2 } \mathrm { d } \mu ( x ) < \infty , \int _ { \mathcal { X } } \| D \mathcal { G } ( x ) \| _ { H S ( \mathcal { X } , \mathcal { Y } ) } ^ { 2 } \mathrm { d } \mu ( x ) < \infty \biggr \} , } \end{array}
$$

where $\boldsymbol { D \mathcal { G } } ( \boldsymbol { x } ) \in \boldsymbol { L } ( \mathcal { X } , \mathcal { Y } )$ is the Fréchet derivative, which we assume to be Hilbert-Schmidt at every $x \in \mathcal { X }$ . Suppose that there exists $C _ { P } > 0$ such that for every d-dimensional subspace $V \subseteq { \mathcal { X } }$ , and every $f \in C ^ { 1 } \cap H ^ { 1 } ( V , ( P _ { V } ) _ { \sharp } \mu ; \mathcal { V } )$

$$
\int _ { V } \| f ( v ) - \mathbb { E } [ f ] \| _ { \mathcal { V } } ^ { 2 } \mathrm { d } \mu ( v ) \leq C _ { P } \int _ { V } \| D f ( v ) \| _ { H S ( V , \mathcal { V } ) } ^ { 2 } \mathrm { d } \mu ( v ) .
$$

This is referred to as a Poincaré inequality. Then

$$
\begin{array} { l } { { C _ { P } \displaystyle \int _ { V } \| D \bar { \mathcal { G } } _ { V } ( v ) \| _ { H S ( V , \mathcal { V } ) } ^ { 2 } \mathrm { d } \mu ( v ) \le C _ { P } \displaystyle \int _ { V } \| \int _ { V ^ { \perp } } D _ { v } \mathcal { G } ( v + w ) \mathrm { d } \mu ( w | v ) \| _ { H S ( V , \mathcal { V } ) } ^ { 2 } \mathrm { d } \mu ( v ) } } \\ { { \displaystyle \qquad \le C _ { P } \displaystyle \int _ { V ^ { \perp } } \| D _ { v } \mathcal { G } ( v + w ) \| _ { H S ( V , \mathcal { V } ) } ^ { 2 } \mathrm { d } \mu ( w | v ) \mathrm { d } \mu ( v ) } } \\ { { \displaystyle \qquad = C _ { P } \displaystyle \int _ { \mathcal { X } } \| D \mathcal { G } ( x ) | _ { V } \| _ { H S ( V , \mathcal { V } ) } ^ { 2 } \mathrm { d } \mu ( x ) } } \end{array}\tag{3.18}
$$

provides an upper bound to the objective in (3.17).

To show how to maximize (3.18), we assume in the following that $\mathcal { G } \in C ^ { 1 } \cap H ^ { 1 } ( \mathcal { X } , \mu ; \mathcal { Y } )$ and define the active subspace covariance operator

$$
\mathcal { C } : = \int _ { \mathcal { X } } D \mathcal { G } ( \boldsymbol { x } ) ^ { * } D \mathcal { G } ( \boldsymbol { x } ) \mathrm { d } \mu ( \boldsymbol { x } ) \in L ( \mathcal { X } , \mathcal { X } ) ,\tag{3.19}
$$

where $D \mathcal { G } ( x ) ^ { * } \in L ( \mathcal { V } , \mathcal { X } )$ is the adjoint of $D \mathcal G ( x )$ . Since at every $x \in \mathcal { X }$ the operator $D \mathcal G ( x )$ is Hilbert-Schmidt (and therefore compact), and $D \mathcal { G } ( x ) ^ { * } \in L ( \mathcal { V } , \mathcal { X } )$ , the operator $D \mathcal { G } ( x ) ^ { \ast } D \mathcal { G } ( x ) \in$ $L ( \mathcal { X } , \mathcal { X } )$ is compact and self-adjoint with $L ( \mathcal { X } , \mathcal { X } )$ -norm bounded by $\| D \mathcal { G } ( x ) \| _ { H S ( \mathcal { X } , \mathcal { V } ) } ^ { 2 }$ . Due to the square integrability of $\| D \mathcal { G } ( x ) \| _ { H S ( \mathcal { X } , \mathcal { V } ) } ^ { 2 } .$ , the operator is well-defined, compact and self-adjoint.

Proposition 3.10. Let V be a d-dimensional subspace of  . Then V maximizes (3.18) if and only if V is a span of eigenvectors of corresponding to the d largest eigenvalues.

Proof. Denote by $e _ { 1 } , \ldots , e _ { d }$ an orthonormal basis of $V \subseteq { \mathcal { X } }$ . Then, by definition of the Hilbert-Schmidt norm

$$
\begin{array} { l } { \displaystyle \int _ { \mathcal { X } } \| D \mathcal { G } ( \boldsymbol { x } ) | _ { V } \| _ { H S ( V , \mathcal { Y } ) } ^ { 2 } \mathrm { d } \mu ( \boldsymbol { x } ) = \displaystyle \int _ { \mathcal { X } } \sum _ { j = 1 } ^ { d } \| D \mathcal { G } ( \boldsymbol { x } ) e _ { j } \| _ { \mathcal { Y } } ^ { 2 } \mathrm { d } \mu ( \boldsymbol { x } ) } \\ { \displaystyle \qquad = \int _ { \mathcal { X } } \sum _ { j = 1 } ^ { d } \langle D \mathcal { G } ( \boldsymbol { x } ) ^ { * } D \mathcal { G } ( \boldsymbol { x } ) e _ { j } , e _ { j } \rangle \mathrm { d } \mu ( \boldsymbol { x } ) } \\ { \displaystyle \qquad = \sum _ { j = 1 } ^ { d } \langle \mathcal { C } e _ { j } , e _ { j } \rangle . } \end{array}
$$

As in the proof of Proposition 3.4, one can show that this quantity is maximized if and only if $\{ e _ { 1 } , \ldots , e _ { d } \}$ is a set of eigenvectors corresponding to the d largest eigenvalues of . □

Remark 3.11 (Error of active subspace approach). Denote by $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \cdots$ the eigenvalues of the active subspace covariance operator  in (3.19). Under suitable additional assumptions, one can show that for V as in Proposition 3.10 and $\bar { \mathcal { G } } _ { V }$ as in (3.16)

$$
\mathbb { E } _ { z \sim \mu } [ \| \mathcal G ( z ) - \bar { \mathcal G } _ { V } ( P _ { V } z ) \| _ { \mathcal { V } } ^ { 2 } ] \le C _ { P } \sum _ { j > d } \lambda _ { j } .
$$

We refer to $I { \boldsymbol { \vartheta } } I _ { ; }$ , Theorem $\it 3 . 4 J$ for the proof of such a statement. This again motivates the maximization of (3.18).

## 3.3.3 Computation of Encoder

The active subspace method can be used to determine eficient linear encoders. Let $( \psi _ { \mathcal { X } , j } ) _ { j = 1 } ^ { d }$ be d orthonormal eigenvectors corresponding to d largest eigenvalues of the active subspace covariance operator $\begin{array} { r } { \mathcal { C } = \int _ { \mathcal { X } } D \mathcal { G } ( x ) ^ { * } D \mathcal { G } ( x ) \mathrm { d } \mu ( x ) } \end{array}$ . An encoder can then be constructed by

$$
\mathcal { E } ( x ) = ( \langle \psi _ { \mathcal { X } , j } , x \rangle ) _ { j = 1 } ^ { d } .
$$

The operator  is unknown, and we have to approximate it from a data sample. A key diference from PCA is that we require information on the operator derivatives. Assume we are given $( D \mathcal { G } ( x _ { j } ) ) _ { j = 1 } ^ { m }$ with $x _ { j } \overset { \mathrm { i i d } } { \sim } \mu$ . The procedure is then as follows:

(i) Compute the empirical active subspace covariance operator

$$
\hat { \mathcal { C } } x = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } D \mathcal { G } ( x _ { j } ) ^ { * } D \mathcal { G } ( x _ { j } ) x \qquad \forall x \in \mathcal { X } .
$$

In a typical computational setting, is a function between two high- but finite-dimensional spaces $\mathbb { R } ^ { k } \to \mathbb { R } ^ { n }$ . In this case $D \mathcal G ( x )$ is represented by a $n \times k$ matrix, and $D \mathcal { G } ( x ) ^ { * } D \mathcal { G } ( x ) =$ $D \mathcal { G } ( x ) ^ { \top } D \mathcal { G } ( x ) \in \mathbb { R } ^ { k \times k }$

(ii) Solve the eigenvalue problem. Compute the eigendecomposition of the symmetric matrix $\hat { \mathcal { C } } \in \mathbb { R } ^ { k \times k }$

$$
\hat { \mathcal { C } } \hat { \psi } _ { \mathcal { X } , j } = \hat { \lambda } _ { j } \hat { \psi } _ { \mathcal { X } , j } .
$$

(iii) Select the active directions. Choose d orthonormal eigenvectors $( \hat { \psi } _ { \mathcal { X } , j } ) _ { j = 1 } ^ { d }$ corresponding to the d largest eigenvalues $( \hat { \lambda } _ { j } ) _ { j = 1 } ^ { d }$ . These vectors form the basis for the empirical active subspace and are used to construct the encoder.

## 3.4 Autoencoders

Autoencoders try to find latent representations of a data distribution $\mu$ on a Banach space $\mathcal { Z } .$ The ansatz is given via the optimization problem

$$
\operatorname* { i n f } _ { A : \mathcal { Z } \to \mathbb { R } ^ { d } } \mathbb { E } _ { z \sim \mu } [ \| z - B \circ \mathcal { A } ( z ) \| _ { \mathcal { Z } } ^ { p } ] ,\tag{3.20}
$$

for some $p \in [ 1 , \infty )$ . Further constraints on and discussed below, but importantly, they are allowed to be nonlinear; as we shall see, restricting and to be afine linear, gives a method that is equivalent to PCA. The interpretation of is that it finds a latent variable representing the most important features of the data. The map reconstructs the input z from its features $\mathcal { A } ( z ) \in \mathbb { R } ^ { d }$ Again we only focus on $p = 2$ and $\mathcal { Z } \mathrm { ~ a ~ }$ Hilbert space.

## 3.4.1 Afine Autoencoder

Suppose that we restrict $A , B$ to be afine linear maps. In this case $B \circ \mathcal { A } ( z ) \in \mathrm { i m } ( B )$ , which is an at most d-dimensional afine subspace of . Denote this subspace by $z _ { 0 } + V$ . The best approximation of z within $z _ { 0 } + V$ is given by $z _ { 0 } + P _ { V } ( z - z _ { 0 } )$ . Therefore, if ,  solve (3.20), it must hold

$$
B \circ A ( z ) = z _ { 0 } + P _ { V } ( z - z _ { 0 } ) .
$$

Assuming $\mu$ to have finite first and second moment, the optimal $z _ { \mathrm { 0 } }$ and $V$ are given by (3.7) and Proposition (3.4). Thus, an optimal solution  and  is given by

$$
\boldsymbol { A } ( z ) = ( \langle z - z _ { 0 } , \psi _ { j } \rangle ) _ { j = 1 } ^ { d } , \qquad \boldsymbol { B } ( ( c ) _ { j = 1 } ^ { d } ) = z _ { 0 } + \sum _ { j = 1 } ^ { d } c _ { j } \psi _ { j }
$$

where $( \psi _ { j } ) _ { j = 1 } ^ { d }$ is an orthonormal eigenbasis corresponding to d largest eigenvalues of the covariance of $\mu$ in (3.9). In this sense (afine) linear autoencoders are equivalent to PCA. We emphasize however that such and are not unique: For any invertible afine transformation $\mathcal { Q } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ , we have that $( B \circ \mathcal { Q } ) \circ ( \mathcal { Q } ^ { - 1 } \circ \mathcal { A } )$ provides another optimal solution to (3.20).

## 3.4.2 Nonlinear Autoencoder

While afine autoencoders can detect optimal linear subspace representations, they fail to capture nonlinear structure in the data. Autoencoders are therefore typically used for $\mathcal { A } , \mathcal { B }$ in nonlinear function classes such as neural networks.

Suppose that $\mu$ is a distribution whose support lies on a compact d-dimensional smooth manifold $M \subseteq { \mathcal { Z } }$ . If M can be parametrized with a single chart $\varphi : [ 0 , 1 ] ^ { d } \to M$ , then any extension $\mathcal { A } : \mathcal { Z }  \mathbb { R } ^ { d }$ of $\varphi ^ { - 1 } : M \to \mathbb { R } ^ { d }$ , and any extension $B : \mathbb { R } ^ { d }  M$ of $\varphi$ satisfy that $B \circ \mathcal { A } ( z ) = z$ for every z in the support of $\mu .$ . Thus the latent dimension d is suficient to perfectly represent the data. In general this is not the case, if M requires multiple charts. However, one can show that the latent dimension $2 d + 1$ still sufices in this case, e.g. [54].

By contrast, linear subspaces may approximate manifolds only poorly, often requiring a much larger latent dimension. In such situations, nonlinear autoencoders can achieve far more eficient representations.

## 3.4.3 Active Subspace Autoencoder

The idea of active subspaces can also be generalized to autoencoders. In this case we replace the objective (3.20) by

$$
\displaystyle \operatorname* { i n f } _ { \mathcal { E } : \mathcal { X } \longrightarrow \mathbb { R } ^ { d _ { \mathcal { X } } } } \mathbb { E } _ { \boldsymbol { x } \sim \boldsymbol { \mu } } \big [ \| \mathcal { G } ( \boldsymbol { x } ) - \bar { \mathcal { G } } \circ \mathcal { E } ( \boldsymbol { x } ) \| _ { \mathcal { X } } ^ { 2 } \big ] .\tag{3.21}
$$

For linear $\mathcal { E } ,$ this corresponds to the active subspace approach. For nonlinear $\mathcal { E } ,$ , the map may learn nonlinear features or a manifold capturing the directions of largest variation of $\mathcal { G }$ . This then yields an encoder map $\mathcal { E }$ for our surrogate.

Note that the objective (3.21) is very generic: training it does not just yield an encoder $\mathcal { E }$ but in principle it can be used to obtain a complete surrogate $\bar { \mathcal { G } } \circ \mathcal { E }$ for $\mathcal { G }$ . In fact, the so-called $\mathrm { \Omega ^ {  } ~ } \mathrm { \Omega } \mathrm { \sim } \mathrm { \varrho } \mathrm { p O N e t } ^ { \mathfrak { N } }$ architecture can be interpreted as an example of that type with a specific structure for $\bar { \mathcal { G } }$ (see Section 4.4 and especially (4.12)).

## 3.4.4 Computation of Encoder and Decoder

Autoencoders can be used to determine nonlinear encoders and decoders $\mathcal { E } ,$ from data as in Section 3. Assume given training data $( x _ { j } , \mathcal { G } ( x _ { j } ) ) _ { j = 1 } ^ { N }$ with $x _ { j } \overset { \mathrm { i i d } } { \sim } \mu$ . To train an encoder $\mathcal { E } : \mathcal { X }  \mathbb { R } ^ { d }$ ， we approximate the objective (3.20) by its Monte-Carlo estimate

$$
\operatorname* { i n f } _ { \varLambda \colon \mathscr { X } \to \mathbb { R } ^ { d } } \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \| x _ { j } - B _ { \mathscr { X } } \circ \mathscr { A } _ { \mathscr { X } } ( x _ { j } ) \| _ { \mathscr { X } } ^ { 2 } .
$$

The minimization can for example be carried out over neural network classes for $\mathcal { A } _ { \mathcal { X } }$ and $B _ { \mathcal { X } }$ . After training, $B _ { \mathcal { X } }$ is discarded, and we obtain an encoder $\mathcal { E } : = \mathcal { A } _ { \mathcal { X } }$ . Alternatively, one may also employ a Monte-Carlo approximation to the active-subspace autoencoder objective (3.21).

To train a decoder, we analogously optimize

$$
\operatorname* { i n f } _ { \ A _ { y : \mathcal { V } \to \mathcal { R } } ^ { m } } \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \| \mathcal { G } ( x _ { j } ) - \mathcal { B } _ { \mathcal { V } } \circ \mathcal { A } _ { \mathcal { V } } ( \mathcal { G } ( x _ { j } ) ) \| _ { \mathcal { V } } ^ { p } .
$$

In this case $\mathcal { A } _ { \mathcal { Y } }$ is discarded, and we set $\mathcal { D } : = B _ { \mathcal { Y } }$

## 4 Architectures

With en- and decoders at hand, we turn to the actual architecture of (families of) finite-parametric operators $\{ \hat { \mathcal { G } ^ { N } } \} _ { N \geq 1 } : \mathcal { X } \to \mathcal { Y }$ acting on infinite-dimensional function spaces. Several classes of architectures of operator surrogates have emerged in the past five years. Most, if not all operator surrogates considered fit into the “Encoder-Approximator-Decoder” paradigm (3.1) i.e. they are of the generic form

$$
\hat { \mathcal G } ^ { N } = { \mathcal D } _ { \mathcal V } \circ g ^ { N } \circ { \mathcal E } _ { { \mathcal X } } \ .\tag{4.1}
$$

Encoders and decoders as discussed above relate function space in- and output to parameter domains in sequence spaces. In addition to finite-parametric surrogate maps $g ^ { N }$ in (4.1), on the input side, dimension-truncation to a finite input dimension $d _ { \mathcal { X } }$ is performed on the coeficient sequence representing the encoded input (signal), and the d<sub>Y</sub>-dimensional output of $g ^ { N }$ is understood to be padded by zeros prior to synthesis operator application of $\mathcal { D } _ { \mathcal { Y } }$ in (4.1). Early examples of mathematical results for (4.1) are [87], and [71].

The approximator $g ^ { N }$ depends on “latent” parameters which are to be found from data by suitable training procedures. Increasing the number N of latent parameters, one minimally aims at some form of universality of the sequence $g ^ { N }$ (see the discussion in Sec. 1.1). Accordingly, the so-called curse of dimensionality must be addressed and, if at all possible, overcome. Severa approaches to obtain suitable $g ^ { N }$ in (4.1) have emerged in the past years: we mention only n-term truncated polynomial chaos expansions and stochastic Galerkin resp. stochastic collocation [45]. neural networks [70], tensor-structured approximation [90], probabilistic “random feature”, kernel surrogates [112, 114], and lattice point interpolation and approximation. We detail here polynomial chaos approximators $g ^ { N }$ in (4.1) giving rise to Spectral Operator Surrogates (with N depending on n). We also discuss several neural network based approximator constructions giving rise to the “DeepONet” and “Neural Operator” surrogates; their structure is distinct from that of (4.1).

## 4.1 Spectral Operators

Consider stable linear representations of the operator in- and output as described in Section 3.1:

$$
x = \sum _ { j \in \mathbb { N } } \varphi _ { \mathcal { X } , j } ( x ) \psi _ { \mathcal { X } , j } \qquad \mathrm { ~ a n d ~ } \qquad y = \sum _ { i \in \mathbb { N } } \varphi _ { \mathcal { Y } , i } ( y ) \psi _ { \mathcal { Y } , i } \qquad \forall x \in \mathcal { X } , \ y \in \mathcal { Y } .\tag{4.2}
$$

An operator $\mathcal { G } : \mathcal { X }  \mathcal { Y }$ can then be decomposed via

$$
\mathcal { G } ( \boldsymbol { x } ) = \sum _ { i \in \mathbb { N } } g _ { i } ( ( \varphi _ { \boldsymbol { \mathscr { X } } , j } ( \boldsymbol { x } ) ) _ { j \in \mathbb { N } } ) \psi _ { \mathscr { V } , i } ,\tag{4.3}
$$

for the functions $g _ { i } : \mathbb { R } ^ { \mathbb { N } }  \mathbb { R }$ . Since the $\varphi _ { \mathcal { X } , j }$ and the $\psi _ { \mathcal { V } , i }$ are known, the task of operator learning now boils down to approximating the $g _ { i }$ . Using (3.2) we have an explicit formula

$$
g _ { i } ( ( \varphi _ { \mathcal { X } , j } ( x ) ) _ { j \in \mathbb { N } } ) = \varphi _ { \mathcal { V } , i } ( \mathcal { G } ( x ) ) = \varphi _ { \mathcal { V } , i } \Bigl ( \mathcal { G } \Bigl ( \sum _ { i \in \mathbb { N } } \varphi _ { \mathcal { X } , j } ( x ) \psi _ { \mathcal { X } , j } \Bigr ) \Bigr ) .\tag{4.4}
$$

If $( \psi _ { \mathcal { V } , j } ) _ { j \in \mathbb { N } }$ is a basis of ${ \mathcal { V } } ,$ , then (4.4) are the unique functions such that (4.3) holds. Due to the linearity of the $\varphi _ { \mathcal { X } , j }$ , the coeficient functions $g _ { i }$ inherit smoothness properties, such as diferentiability, from the operator ${ \mathcal { G } } .$ . Since many practically relevant operators are highly smooth, this motivates the use of spectral methods to approximate the $g _ { i }$

## 4.1.1 Polynomial Chaos Expansions

Each coeficient function $g _ { i }$ is a function of the infinitely many variables $\bar { \boldsymbol { x } } _ { j } : = \varphi _ { \boldsymbol { \mathcal { X } } , j } ( \boldsymbol { x } ) \in \mathbb { R } , j \in \mathbb { N }$ We also write

$$
\bar { \pmb { x } } = ( \bar { x } _ { j } ) _ { j \in \mathbb { N } } = ( \varphi _ { \mathcal { X } , j } ( \boldsymbol { x } ) ) _ { j \in \mathbb { N } } \in \mathbb { R } ^ { \mathbb { N } } .\tag{4.5}
$$

Under certain conditions, such infinite-parametric functions allow for multivariate polynomial expansions (known as “polynomial chaos”), see, e.g., [152, 48] or [156, Section 2.1]. A simple example is Taylor polynomial chaos: Denote

$$
\mathcal { F } : = \left. \nu = ( \nu _ { j } ) _ { j \in \mathbb { N } } : \nu _ { j } \in \mathbb { N } _ { 0 } , \sum _ { j \in \mathbb { N } } \nu _ { j } < \infty \right. .
$$

The Taylor polynomial chaos expansion of $g _ { i }$ (assuming it exists) then reads

$$
g _ { i } ( ( \varphi \chi , \mathfrak { j } ( x ) ) _ { j \in \mathbb { N } } ) = g _ { i } ( \bar { \pmb x } ) = \sum _ { \pmb { \nu } \in \mathcal { F } } c _ { i , \pmb { \nu } } \bar { \pmb x } ^ { \pmb \nu } ,
$$

for certain coeficients $c _ { i , \nu } \in \mathbb { R }$ . Under suitable assumptions it holds that $\begin{array} { r } { c _ { i , \nu } = \frac { 1 } { \nu ! } \frac { \partial ^ { \nu } } { \partial x ^ { \nu } } g _ { i } ( \pmb { x } ) | _ { \pmb { x } = \mathbf { 0 } } , \mathrm { e . g . } } \end{array}$ [156, Lemma 2.1.2]; here $\begin{array} { r } { \pmb { \nu } ! = \prod _ { j \in \mathrm { s u p p } ( \pmb { \nu } ) } \nu _ { j } ! } \end{array}$ and $\begin{array} { r } { \pmb { x } ^ { \nu } = \prod _ { j \in \mathrm { s u p p } ( \pmb { \nu } ) } x _ { j } ^ { \nu _ { j } } } \end{array}$ where supp $( \nu ) = \{ i : \nu _ { i } \neq 0 \}$ and empty products equal 1 by convention.

For a finite set of multiindices $\Lambda _ { i } \subseteq \mathcal { F }$ , the ith coeficient function $g _ { i }$ can then be approximated by a truncated polynomial chaos expansion

$$
g _ { i } ( ( \varphi \chi , \mathfrak { j } ( x ) ) _ { \mathfrak { j } \in \mathbb { N } } ) \simeq \sum _ { \pmb { \nu } \in \Lambda _ { i } } c _ { i , \pmb { \nu } } \bar { \pmb { x } } ^ { \pmb { \nu } } .
$$

This yields an approximation to the operator via

$$
\mathcal { G } ( \boldsymbol { x } ) \simeq \sum _ { i \in \mathbb { N } } \psi _ { \mathcal { V } , i } \sum _ { \nu \in \Lambda _ { i } } c _ { i , \nu } \bar { \boldsymbol { x } } ^ { \nu } .\tag{4.6}
$$

In order for this formula to be computable, we require $\Lambda _ { i } = \varnothing$ for all but finitely many i. Since the right-hand side in (4.6) is (afine) linear in the unknown coeficients $\displaystyle c _ { i , \nu }$ , they can be fitted by least-squares. In practice, for stability reasons other expansions such as multivariate Legendre or Hermite polynomials are preferred.

A key dificulty of this approach is the choice of eficient multiindex sets $\Lambda _ { i }$ . A concrete construction is provided in [71, Sections 5 and 6]. Here we only give some intuition:

• $g _ { i }$ is the coeficient function corresponding to $\psi _ { \mathcal { V } , i } \in \mathcal { V }$ on the output side. The representation system on the output side is typically ordered from coarse to fine scales. This is for instance the case if $\psi _ { \mathcal { V } , i }$ is a Fourier type basis or a wavelet, e.g. $\psi _ { \mathcal { V } , i } ( \eta ) = \sin ( i \eta ) , \eta \in \mathbb { R }$ . Assuming smoothness of the operator output, low-index (small i) functions capture the dominant features of the output, while high-index functions correspond to finer details. As smoothness assumptions imply faster decay of the coeficient sequence, in this case one expects the cardinality of the $\Lambda _ { i }$ to decrease, i.e.

$$
| \Lambda _ { 1 } | \geq | \Lambda _ { 2 } | \geq \ldots .
$$

• Similar considerations hold for the representation of the input with respect to $\psi _ { \mathcal { X } , j }$ , cf. (4.2). Since lower $j$ corresponds to more dominant features of the input, for fixed $i ,$ the multiindex set $\Lambda _ { i } \subseteq \mathcal { F }$ should be anisotropic, with higher importance given to lower $j .$

Without recalling the full details of [71], the multiindex sets constructed there are of the following type, which reflects the above intuition: For two null sequences $\pmb { \rho } = ( \rho _ { j } ) _ { j \in \mathbb { N } } , \pmb { \xi } = ( \xi _ { i } ) _ { i \in \mathbb { N } }$ with values in (0, 1),

$$
\Lambda _ { i } = \{ \pmb { \nu } \in \mathscr { F } : \xi _ { i } \pmb { \rho } ^ { \nu } \geq \varepsilon \} ,
$$

with $\varepsilon > 0$ being a parameter associated to the overall accuracy of the scheme. This construction ensures in particular $\Lambda _ { i } = \varnothing$ for all but finitely many $i .$

## 4.1.2 Sparse-Grid Interpolation

A closely related approach is to use sparse-grid interpolants instead of polynomial chaos expansions. With the notation in (4.5), the surrogate is constructed as

$$
\mathcal { G } ( x ) \simeq \sum _ { i \in \mathbb { N } } I _ { \Lambda _ { i } } [ g _ { i } ] ( \bar { \pmb x } ) \psi _ { \mathcal { V } , i } ,\tag{4.7}
$$

where $I _ { \Lambda _ { i } } [ g _ { i } ]$ is a sparse-grid interpolant of $_ { g _ { i } ; }$ we recall its construction below. The $\Lambda _ { i } \subseteq { \mathcal { F } }$ are again multiindex sets, for which the remarks in Section 4.1.1 hold verbatim. One advantage of this approach is that it has deterministic convergence guarantees, see [71].

Computing (4.7) requires computation of the interpolation operators. We assume in the following that each $\Lambda _ { i }$ is downward closed, meaning that if $\nu \in \Lambda _ { i }$ , then $\pmb { \mu } \in \Lambda _ { i }$ for all $\pmb { \mu } \in \mathcal { F }$ such that $\mu _ { i } \leq \nu _ { i }$ for all $i \in \mathbb N$ . For $n \in \mathbb { N }$ denote by $I _ { n } : C ^ { 0 } ( \mathbb { R } ) \to \mathbb { P } _ { n }$ a univariate polynomial interpolant in $n + 1$ points; typical examples are Chebyšev or Hermite interpolants. With the multivariate tensorized interpolants

$$
I _ { \nu } = \otimes _ { j \in \mathrm { s u p p } \nu } I _ { \nu _ { j } }
$$

the corresponding sparse-grid interpolant of coeficient map $g _ { i }$ is defined as

$$
I _ { \Lambda _ { i } } [ g _ { i } ] = \sum _ { \pmb { \nu } \in \Lambda _ { i } } a _ { \pmb { \nu } , i } I _ { \pmb { \nu } } [ g _ { i } ] \qquad \mathrm { w h e r e } \qquad a _ { \pmb { \nu } , i } = \sum _ { \pmb { e } \in \{ 0 , 1 \} ^ { \mathbb { N } } } ( - 1 ) ^ { | \pmb { e } | } ,
$$

where $\begin{array} { r } { | e | = \sum _ { j \in \mathbb { N } } e _ { j } \in \mathbb { N } _ { 0 } } \end{array}$ . For more details and explicit formulas we refer for example to [16, 76, 156].

We emphasize that the computation of (4.7) requires us to evaluate the coeficient maps $g _ { i }$ in (4.4) at the interpolation points, which are determined by the one-dimensional interpolation operators $I _ { n } , n \in \mathbb { N } ,$ , as well as the sets $\Lambda _ { i } , \textit { i } \in \mathbb { N }$ . As long as the $\Lambda _ { i }$ are fixed, so is the set of required evaluation points of the operator. These operator evaluations can then be done a priori and in parallel. Subsequently, no further form of training is required, as the surrogate has an explicit formula.

\- We apply operator surrogates based on sparse-grid interpolation to the parametric difusion problem in notebook ${ 2 _ { - } }$ Interpolation.ipynb.

## 4.1.3 Low-rank Tensor Approximations

Another possibility is to represent the $g _ { i }$ via a low-rank representation of its values at some grid. Specifically, fix a set of univariate interpolation points $\xi _ { 0 } , \ldots , \xi _ { n } \in \mathbb { R }$ ; these could for example again be Chebyšev or Gauss-Hermite points. Setting the variables after the first d dimensions to $0 , g _ { i }$ can be approximated by the tensor interpolant

$$
I _ { n } \otimes I _ { n } \otimes \cdot \cdot \cdot \otimes I _ { n } \big [ g _ { i } \big ( \cdot , \cdot \cdot , \cdot , 0 , 0 , . . . \big ) \big ] .\tag{4.8}
$$

This representation requires us to store the function values at the interpolation points, i.e. the d tensor

$$
g _ { i } ( \xi _ { j _ { 1 } } , \dots , \xi _ { j _ { d } } ) _ { j _ { 1 } , \dots , j _ { d } = 0 } ^ { n } \in \mathbb { R } ^ { ( n + 1 ) \times \cdots \times ( n + 1 ) } .\tag{4.9}
$$

The idea is to compress this tensor via a low rank representation.

A popular low-rank tensor format is the tensor train (TT) format [120]. It tries to approximate (4.9) via

$$
g _ { i } ( \xi _ { j _ { 1 } } , \dots , \xi _ { j _ { d } } ) \simeq { \pmb v } ^ { ( 1 ) } ( j _ { 1 } ) \cdots { \pmb v } ^ { ( d ) } ( j _ { d } ) \qquad \forall j _ { 1 } , \dots , j _ { d } = 0 , \dots , n ,\tag{4.10}
$$

with the TT cores

$$
\begin{array} { r l r l r } { v ^ { ( 1 ) } \in \mathbb { R } ^ { 1 \times ( n + 1 ) \times r } , \ v ^ { ( d ) } \in \mathbb { R } ^ { r \times ( n + 1 ) \times 1 } } & { } & { \mathrm { a n d } } & { } & { v ^ { ( j ) } \in \mathbb { R } ^ { r \times ( n + 1 ) \times r } \ \forall j = 2 , \ldots , d - 1 . } \end{array}
$$

The memory cost of this format is $O ( r ^ { 2 } n d )$ , compared to $O ( n ^ { d } )$ for the direct storage of (4.9). For moderate ranks $^ { r , }$ this is a significant improvement. Moreover, given a representation in the TT format, the interpolant in (4.8) is very eficient to evaluate: Due to the separation of the cores (4.10), the interpolation can be done individually in each dimension. Evaluating the interpolant thus scales only linearly in the dimension.

A key question is how to eficiently obtain a TT approximation of the tensor in (4.9). A standard algorithm is so-called TT-cross [119]. A drawback is that it requires many sequential evaluations of ${ \mathcal { G } } ,$ which can make the process slow in practice. For this reason the ALS-cross algorithm was developed in the context of parametric PDEs [43]; while it also relies on sequential evaluations of ${ \mathcal { G } } ,$ it can be significantly more eficient. The downside is that it requires a specific structure for the operator ${ \mathcal { G } } ,$ thereby limiting its applicability.

## 4.2 Nonlocal Operator Networks

A unifying mathematical framework covering a wide class of operator surrogate architectures (4.1) was recently promulgated in [79] under the label “Neural Operator”. In order to distinguish it from other surrogate architectures based on neural networks we here refer to this architecture as Nonlocal Operator Network (Nonlocal ONet). Nonlocal ONets mimic the architectures of traditional finite-parametric, deep feedforward neural networks in finite dimensions, while $\mathrm { a ) }$ admitting infinitedimensional input and output spaces and , b) replacing the afine weight-bias maps by linear operators, and, importantly, by application of a nonlocal (integral) operator with prescribed kernel (“linear kernel layer” [79], “continuum attention” [17] ) and c) by injecting nonlinearity, as in the finite-dimensional case, via activation functions. Acting on function spaces, these are interpreted either in a pointwise sense, or as Nemytskii operators in the (Sobolev or Besov) function spaces of interest.

The general structure of a nonlocal ONet of finite depth $L \in \mathbb { N }$ acting between Banach input space  and output space $\mathcal { V }$ is [79]

$$
\hat { \mathcal { G } } : \mathcal { X } \to \mathcal { Y } : u \mapsto Q \circ \mathcal { L } _ { L } \circ . . . \circ \mathcal { L } _ { 2 } \circ \mathcal { L } _ { 1 } \circ \mathcal { R } ( u ) \ .\tag{4.11}
$$

The layers $\mathcal { L } _ { \ell } : \mathcal { X } _ { \ell }  \mathcal { X } _ { \ell + 1 } , \ell = 1 , . . . , L$ , are nonlinear maps between suitable function spaces $\{ \mathcal { X } _ { \ell } \} _ { \ell = 1 , \dots , L + 1 }$ of the general form

$$
\mathcal { L } _ { \ell } : \mathcal { X } _ { \ell }  \mathcal { X } _ { \ell + 1 } : v \mapsto \sigma ( W _ { \ell } v + b _ { \ell } + K _ { \ell } ( v ; \gamma _ { \ell } ) ) , \quad \ell = 1 , 2 , . . . , L .
$$

Here, the activation function $\sigma ( \cdot )$ is understood to act in a pointwise sense in $\big [ 7 9 \big ]$ , which mandates that $\mathcal { X } _ { \ell } \subset C ^ { 0 } ( \mathrm { D } _ { \ell } )$ for some domain $\mathrm { D } _ { \ell } \subset \mathbb { R } ^ { d _ { \ell } } , d _ { \ell } \in \mathbb { N }$ . Further, $W _ { \ell } \in L ( \mathcal { X } _ { \ell } , \mathcal { X } _ { \ell + 1 } )$ is a bounded, linear operator, $b _ { \ell } \in \mathcal { X } _ { \ell + 1 }$ a bias term, and $\displaystyle \kappa _ { \ell }$ a nonlocal integral operator of the generic form

$$
\mathcal { K } _ { \ell } ( v ; \gamma ) ( \eta ) : = \int _ { \zeta \in \mathrm { D } _ { \ell } } \kappa _ { \ell } ( \eta , \zeta ; \gamma ) v ( \zeta ) d \zeta , \quad \eta \in \mathrm { D } _ { \ell } ,
$$

with suitable kernel functions $\kappa _ { \ell } ( \cdot , \cdot ; \gamma )$ which depend on (trainable) latent parameter vectors $\gamma .$ . In (4.11), $\mathcal { R } : \mathcal { X }  \mathcal { X } _ { 1 }$ and $\mathcal { Q } : \mathcal { X } _ { L + 1 }  \mathcal { Y }$ . All layers $\mathcal { L } _ { \ell }$ in (4.11) may share the same kernel, in which the subscript ℓ is dropped in the definition of $\mathcal { L } _ { \ell } , \mathrm { i . e . } \mathrm { D } _ { \ell } = \mathrm { D } , \mathcal { X } _ { \ell } = \mathcal { X }$ and $\kappa _ { \ell } = \kappa$ for $\ell = 1 , . . . , L$

In (4.11), the kernel parameters are (a subset of) trainable parameters of the neural operator ${ \hat { \mathcal { G } } } ,$ i.e. $\{ \gamma _ { \ell } \} _ { \ell = 0 , 1 , \dots , L } \subset \theta$ . Definition (4.11) as such is not finite-parametric, due to it involving operators acting on the function spaces $\mathcal { X } _ { \ell }$ . A nonlocal ONet as in (4.11) could be viewed as “infinite-width closure” or continuum limit of finite-parametric, fully connected feedforward networks (subject to suitable scaling hypotheses).

Computationally viable architectures arise from nonlocal ONets (4.11) via discretization, where any discretization method (Nyström, Galerkin, quadrature,...) could be employed. The conceptual abstraction (4.11) in the form of nonlocal ONets allows, however, by judicious choices of discretizations, to unify existing constructions and to generate new, finite-parametric architectures. In any case, a discretization error analysis is required to quantify the diference between a nonlocal ONet and its computational version, see, e.g. [78, 89] for a study for Fourier Neural operators (FNO).

Particular choices of the kernels $\kappa _ { \ell }$ in the operators $\kappa _ { \ell }$ will give rise to particular neural operators: translation-invariant kernels $\kappa ( \eta , \zeta ; \gamma ) = \bar { \kappa } ( z ; \gamma ) \ | _ { z = \eta - \zeta }$ yield the convolutional neural operator [125]. It can be discretized and numerically realized via discrete Fourier- and inverse Fourier transforms [78, 89], Fast Multipole Methods (FMMs) [94], -Matrix Techniques, Tensor-structured Formats [90], to name but a few. In periodic settings, where $\mathcal { X } _ { \ell }$ are function spaces on the torus $\mathbb { T } ^ { d }$ with periodicity, (4.11) takes the form of the Fourier Neural Operator [78, 79], i.e.

$$
\mathcal { K } _ { \ell } ( v ; \gamma ) ( x ) : = \mathcal { F } _ { \xi  \eta } ^ { - 1 } ( \hat { \kappa } _ { \ell } ( \xi ; \gamma ) \hat { v } ( \xi ) )
$$

with $\hat { \kappa } _ { \ell }$ denoting the Fourier-transform (“Fourier-symbol”) of translation-invariant kernel $\bar { \kappa } _ { \ell } ( z ; \gamma )$ i.e. $\hat { \kappa } _ { \ell } ( \xi ; \gamma ) = \mathcal { F } _ { z  \xi } \bar { \kappa } _ { \ell } ( z ; \gamma )$

\- An implementation of the discretized FNO for one-dimensional input and output functions is provided and applied to the parametric difusion problem in notebook 4\_FNO.ipynb.

## 4.3 PCA-Net and FrameNet

The PCA-Net architecture was among the first neural network-based approaches for learning operator surrogates [72, 13]. It fits the abstract encoder-approximator-decoder framework (3.1). Given data $( x _ { i } , \mathcal { G } ( x _ { i } ) ) _ { i = 1 } ^ { m }$ with $x _ { i } \overset { \mathrm { i i d } } { \sim } \mu$ , the training is in two steps: first, the encoder $\mathcal { E } : \mathcal { X }  \mathbb { R } ^ { d _ { \mathcal { X } } }$ and decoder $\mathcal { D } : \mathbb { R } ^ { d y }  \mathcal { V }$ are constructed via empirical PCA (see Section 3.2). Second, the approximator $g : \mathbb { R } ^ { d _ { \mathcal { X } } }  \mathbb { R } ^ { d _ { \mathcal { Y } } }$ in (3.1) is set up as a feedforward neural network, and is trained on the resulting latent variables. The error analysis for empirical PCA is well-developed (see, e.g., [129, 108]). In the specific context of PCA-Net, an analysis was done in [84]: due to data randomness, the error bounds from the PCA reconstruction errors scale in general only with the inverse square root of the number of data points, i.e., the Monte-Carlo rate; this dimension reduction error is independent of the choice of coeficient map, and is thus a general limitation valid also for other PCA-based architectures.

Closely related is the FrameNet architecture [71, 128], where the encoder and decoder are defined by a priori (data-independent) Banach frames on  and , see Section 3.1. For PDE-based solution operators, this is a feasible approach, as suitable frame constructions for typical solution spaces are well-known in the literature, e.g., Example 3.3. While this allows for fast algebraic convergence rates for the approximation error [71], this advantage does not necessarily carry over to the generalization error, see Section 7 and also [61].

In practice, while FrameNet has desirable theoretical properties, PCA-Net is often simpler to implement. Moreover, by using data-dependent representations for its input and output spaces, PCA-Net may be better able to capture latent structures within the operator  and the input measure $\mu .$

\- An implementation of PCA-Net is provided and applied to the parametric difusion problem in notebook 3\_PCANet.ipynb.

## 4.4 DeepONet

The “DeepONet” class of neural operators is one of the earliest approaches. Its “branch-and-trunk” architecture difers at first sight from (3.1) considered elsewhere in this text, but can also be interpreted in this sense, see below. Universality of the DeepONet architectures has first been established in [21], on compact subsets of admissible input signals, and in [87] also in the non-compact case. There, also first error bounds which are free from the CoD were obtained, in the case of holomorphic maps, and with PCA decoding.

We refer to [100] for a succinct presentation of DeepONet architecture, comparison with Fourier-Neural Operator architecture, and quantitative comparisons. A key feature of DeepONet is the socalled “branch Net” part of the DeepONet architecture (see [100, Fig. 1] for an illustration), which performs decoding of signals in a feature space which is data-adaptive, i.e. learned from training data. Importantly, and contrary to the PCA approach, this decoder is learned simultaneously with the approximator.

The trunk-net part of the DeepONet takes the coordinates η in the physical domain D which supports the output, and the branch net takes the discretized function x as input. The output of

the DeepONet takes the generic form

$$
\hat { \mathcal { G } } ( x ) ( \eta ) = \sum _ { k = 1 } ^ { d _ { \mathcal { X } } } b _ { k } ( x ) t _ { k } ( \eta ) + b _ { 0 } , \quad \eta \in \mathrm { D }\tag{4.12}
$$

where $b _ { 0 } ~ \in ~ \mathbb { R }$ is a bias, $\{ b _ { 1 } , b _ { 2 } , . . . , b _ { d _ { \mathcal { X } } } \}$ are $d _ { \mathcal { X } }$ outputs of the branch net, and $\{ t _ { 1 } , t _ { 2 } , . . . , t _ { d _ { \chi } } \}$ representing the DeepONet output are the $d _ { \mathcal { X } }$ output features of the trunk net, which can be interpreted as an encoder $\mathcal { E } : \mathcal { X }  \mathbb { R } ^ { d _ { \mathcal { X } } }$ in (3.1); also see Section 3.4.3. The approximator and decoder $\mathcal { D } \circ g : \mathbb { R } ^ { d _ { \mathcal { X } } }  \mathcal { Y }$ then correspond to the map $\begin{array} { r } { ( c _ { j } ) _ { j = 1 } ^ { d _ { \mathcal { X } } } \mapsto b _ { 0 } + \sum _ { k = 1 } ^ { d _ { \mathcal { X } } } c _ { k } t _ { k } ( \eta ) } \end{array}$

Remark 4.1. We may interpret the DeepONet’s components within an encoder-decoder framework by defining the branch net as the encoder. While not canonical, this choice aligns with our presentation of active subspace autoencoders in Section 3.4.3.

DeepONet is a neural operator architecture framework which, a-priori, does not restrict the branch net and the trunk net yet to any specific architecture. As $\xi \in \mathrm { D }$ is the physical domain of the input function, a feedforward NN is often used as the trunk net. The architecture of the branch net which defined the output feature space can be chosen as a convolutional neural network (CNN), ResNet, or a graph neural network (GNN), etc. For example, if the discretization of x is equispaced and cartesian, a CNN or a feedforward NN ofers advantages. If the discretization of x is on an unstructured mesh, then a GNN is preferable.

## 4.5 Transformers

Starting with the seminal [148] introduction of transformers for finite-dimensional maps, transformer architectures have emerged in recent years as key ingredients for large language models, such as Chat-GPT. They are also at the heart of integrating multimodal data (such as images, text and sound) into large language models [17, 10]. Their current relevance for SciML and SciAI lies in transformer ability to learn from and generalize across multimodal PDE input data (constitutive equations, domain shapes, loadings, initial and boundary data, solution snapshots, form of the PDE, etc.). For operator learning in a PDE context, this capability is key for Scientific AI: the generalization of data-to-solution maps across a range of partial diferential equations [69], e.g., from a simple difusion equation in a square to linear elastostatics in polygonal domains. Multimodal inputs can also arise in the challenging task of generalizing across physical domains, or of identifying a PDE from data and snapshot solutions, or of the inverse problems to identify a domain of a PDE from solution snapshots.

An early contribution to transformer-architectures for neural PDE solvers in SciML is [19].

More recently, on the side of mathematical theory, universal approximation of transformerbased neural operators has been shown e.g. for stochastic processes in path space in [1] and in the finite-dimensional setting in [55], and references there. See also [150] for Transformer-generalization of PDE solution operators across classes of physical domains. The recent work [109] leverages transformers for prediction of statistical properties of (turbulent) flows. Approximation rate bounds for transformer-based neural architectures have been established in [55, 136].

To present the transformer architectures $\hat { \mathcal { G } } _ { T }$ in function space, we follow [17, Sec. 4] and place ourselves again in an encoder-transformer-decoder architecture, i.e.

$$
\hat { \mathcal { G } } _ { T } : \mathcal { X } \to \mathcal { V } : x \mapsto \mathrm { T } _ { o u t } \circ \mathrm { E } _ { L } \circ \mathrm { T } _ { i n } ( x ) ~ .\tag{4.13}
$$

Here, $\mathrm { T } _ { i n } : \mathcal { X }  \mathcal { V }$ maps input data $x \in \mathcal { X }$ to an embedding function space $\nu$ which could be viewed as latent function space. Importantly, $\mathrm { T } _ { i n }$ which is key for the transformers’ ability to deal with multimodal data is not a linear operator as $\mathrm { e . g . }$ ., in PCA-Net and FrameNet. The operator $\mathrm { E } _ { L } : \mathcal { V } \to \mathcal { V }$ is the nonlinear, function space analog to the transformer encoder block in [148], and $\mathrm { T } _ { o u t } : \mathcal { V }  \mathcal { V }$ is a linear decoder. To describe the structure of the blocks $\mathrm { T } _ { i n } , \mathrm { E } _ { L } , \mathrm { T } _ { o u t }$ in (4.13) in more detail, we denote $\mathrm { ~ D ~ } \subset \mathbb { R } ^ { d _ { \eta } }$ and $\mathbf { D } ^ { \prime } \subset \mathbb { R } ^ { d _ { \zeta } }$ bounded domains on which the input and output function spaces, $\mathcal { X } = \mathcal { X } ( \mathrm { D } ; \mathbb { R } ^ { d _ { \boldsymbol { x } } } )$ and $\mathcal { Y } = \mathcal { Y } ( \mathrm { D } ^ { \prime } ; \mathbb { R } ^ { d _ { \boldsymbol { y } } } )$ are modelled, respectively (think of Besov or Sobolev spaces on D resp. $\mathrm { D ^ { \prime } }$ , of vector-valued functions), and the map to be emulated being $\mathcal G : \mathcal X \to \mathcal y : a \mapsto u$

In constructing $\mathrm { T } _ { i n }$ , an input $x \in \mathcal { X } ( \mathrm { D } ; \mathbb { R } ^ { d _ { x } } )$ is positionally encoded by appending the spatial coordinate $\eta$ to each function value:

$$
x _ { i n } ( \eta ) : = ( x ( \eta ) , \eta ) , \quad \eta \in \mathrm { D } .
$$

This approach breaks the translation equivariance of the learned weights and biases, and is also common in Fourier Neural Operators.

Subsequently, a linear transformation $W _ { i n } \in \mathbb { R } ^ { d _ { m o d e l } \times ( d _ { x } + d _ { \eta } ) }$ is applied to $x _ { i n } ( \eta )$ for $\eta \in \mathrm { D }$ to lift $x _ { i n }$ to $\nu ,$ so that

$$
( \mathrm { T } _ { i n } ( x ) ) ( \eta ) : = W _ { i n } x _ { i n } ( \eta ) , \quad \eta \in \mathrm { D } .
$$

The transformation $\mathrm { T } _ { i n }$ acts locally, pointwise $( \mathrm { w . r . t . } \ \eta \in \mathrm { D } )$ on $x _ { i n }$ , with output in the latent embedding space $\mathcal { V } ( \mathrm { D } , \mathbb { R } ^ { d _ { m o d e l } } )$

The transformer block $\mathrm { E } _ { L } : \mathcal { V } \to \mathcal { V }$ in (4.13) acts on inputs $v ^ { ( 0 ) } : = \mathrm { T } _ { i n } ( x ) \in \mathcal { V }$ by an iteration (reminiscent of recurrent blocks in operator unrolling) which is defined, for $l = 1 , 2 , . . . , L$ , by the formal assignments

$$
\begin{array} { r c l } { { v ^ { ( l - 1 ) } } } & { {  } } & { { W _ { 1 } ^ { ( l - 1 ) } v ^ { ( l - 1 ) } + \mathrm { A } _ { m u l t i h e a d } ^ { ( l - 1 ) } ( v ^ { ( l - 1 ) } ) , } } \\ { { v ^ { ( l - 1 ) } } } & { {  } } & { { \mathrm { F } _ { L a y e r N o r m } ^ { ( l - 1 ) } ( v ^ { ( l - 1 ) } ) , } } \\ { { v ^ { ( l - 1 ) } } } & { {  } } & { { W _ { 2 } ^ { ( l - 1 ) } v ^ { ( l - 1 ) } + \mathrm { F } _ { N N } { } ^ { ( l - 1 ) } ( v ^ { ( l - 1 ) } ) , } } \\ { { v ^ { ( l ) } } } & { {  } } & { { \mathrm { F } _ { L a y e r N o r m } ^ { ( l - 1 ) } ( v ^ { ( l - 1 ) } ) . } } \end{array}\tag{4.14}
$$

with $\mathrm { E } _ { L } ( v ^ { ( 0 ) } ) : = v ^ { ( L ) } \in \mathcal { V }$ Here, the nonlinear $\mathrm { F } _ { L a y e r N o r m }$ performs layer normalization, and $\mathrm { F } _ { N N } : \mathcal { V }  \mathcal { V }$ is a standard feedforward neural network layer.

In (4.14), the multi-head, self-attention operator $\mathrm { A } _ { m u l t i h e a d } : \mathcal { V }  \mathcal { V }$ is a composition of a (multi-) linear transformation $\mathrm { W } _ { m u l t i h e a d }$ of $\mathrm { N } _ { s a } \in \mathbb { N }$ many self-attention operators,

$$
( \mathrm { A } _ { m u l t i h e a d } ( v ) ) ( \eta ) = \mathrm { W } _ { m u l t i h e a d } ( A _ { * } ( v ; \theta _ { 1 } ) ( \eta ) , . . . , A _ { * } ( v ; \theta _ { \mathrm { N } _ { s a } } ) ( \eta ) ) , \eta \in \mathrm { D } ,
$$

with $A _ { * }$ a trainable (with training parameters $\theta _ { j } )$ continuum self-attention operator acting pointwise on inputs $v \in \mathcal { V } ( \mathrm { D } , \mathbb { R } ^ { d _ { m o d e l } } )$

The continuum self-attention operator is obtained in [17, Appendix 1] by a limiting argument from the originally proposed discrete self-attention in [148]. With the probability density function $p ( \cdot ; v , \eta ) : \mathrm { D }  \mathbb { R } _ { + }$

$$
p ( \zeta ; v , \eta ) = \frac { \exp ( \langle Q v ( \eta ) , K v ( \zeta ) \rangle ) } { \int _ { \mathrm { D } } \exp ( \langle Q v ( \eta ) , K v ( s ) \rangle ) d s } , \quad \eta \in \mathrm { D } , \zeta \in \mathrm { D } ,
$$

where $Q , K$ are query and key matrices, respectively, it takes the form

$$
A _ { * } ( v ) ( \eta ) = \mathbb { E } _ { \zeta \sim p ( \zeta ; v , \eta ) } [ V v ( \zeta ) ] , \quad \eta \in \mathrm { D } .
$$

Here, V is a (trainable) value matrix.

The weights $W _ { 1 } ^ { ( l - 1 ) } , W _ { 2 } ^ { ( l - 1 ) }$ in (4.14) are linear (matrix-multiplication) operations that act pointwise: $v ( \eta ) \mapsto W _ { i } ^ { ( l - 1 ) } v ( \eta )$ . Similar integral-operator representations can be obtained for a “continuum cross-attention” operator, see [17, Appendix 2] for details.

An important observation is that for $Q , K = 0$ and $V = I .$ , the continuum self-attention map reduces to a simple averaging, which can be viewed a continuous analog to the pooling operation:

$$
A ( v ; Q , K , V ) ( \eta ) = \mathbb { E } _ { \zeta \sim p ( \zeta ; v , \eta ) } [ V v ( \zeta ) ] = \frac { 1 } { | \mathrm { D } | } \int _ { \mathrm { D } } v ( \zeta ) d \zeta .
$$

This implies universal approximation of transformers [17, Theorems 22,23], as a consequence of the general result in [86].

## 5 Training

Constructing a surrogate model for a function $\mathcal { G }$ is typically framed as a supervised learning problem. This process relies on a training dataset consisting of $m \in \mathbb { N }$ input-output pairs of the form

$$
( x _ { i } , y _ { i } ) _ { i = 1 } ^ { m } \qquad \mathrm { w h e r e } \qquad y _ { i } = \mathcal { G } ( x _ { i } ) + \varepsilon _ { i } \in \mathcal { V } .\tag{5.1}
$$

The goal is to learn an approximation of $\mathcal { G }$ from this data. Again, $x _ { i }$ represent the input points, and the $\varepsilon _ { i }$ may account for (measurement) noise or numerical errors, depending on the source of the data. The training points $x _ { i }$ are assumed to be i.i.d. draws from a probability measure $\mu$ on $\mathcal { X } .$ Alternatively, they can be a priori selected based on a fixed design strategy, or chosen adaptively to maximize the information gained about , which is known as active learning.

Remark 5.1. A common modelling assumption is that $\varepsilon _ { i }$ is white noise. In this case $y _ { i }$ is formally not an element $o f { \mathcal { Y } }$ . We omit discussing this in more detail, and refer to [128].

We distinguish between three cases:

• direct construction: In some cases, the surrogate can be constructed directly without the need for optimization. This applies when there is a known, computable map from the data to the surrogate function, as in classical interpolation methods (e.g., the sparse-grid spectral operator in Section 4.1.2, or kernel based approaches [11] computed by directly solving a linear system). These methods typically require the ability to evaluate $\mathcal { G }$ at specific, deterministically chosen points.

• convex objective: For some models, the training objective is a convex function of the parameters θ. This typically occurs when the surrogate $\mathcal { G } _ { \theta }$ depends afine linearly on θ and a standard loss function is used. The resulting optimization problem usually has at least one global minimum, which can be found eficiently. Examples include polynomial chaos surrogates (Section 4.1.1), compressed sensing based approaches [2], random feature models [112], and kernel-based methods [11].

• non-convex objective: This is the most common scenario in operator learning, particularly when deep neural networks are part of the architecture. Here, the surrogate $\mathcal { G } _ { \theta }$ depends nonlinearly on its parameters $\theta ,$ yielding a non-convex optimization landscape with multiple local minima. This includes the majority of deep learning-based architectures.

For simplicity, we let $\mathcal { X } , \mathcal { y }$ be Hilbert spaces in the rest of this section.

## 5.1 Empirical Risk Minimization

A typical loss function used in operator learning is the square loss. For a class of operator surrogates $\hat { \mathcal { G } } _ { \theta }$ depending on a finite dimensional parameter θ, given data (5.1) we then wish to minimize the empirical risk

$$
\hat { \mathcal { R } } _ { m } ( \theta ) : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \| \hat { \mathcal { G } } _ { \theta } ( x _ { j } ) - y _ { j } \| _ { \mathcal { V } } ^ { 2 }\tag{5.2}
$$

in the parameter θ. In practice the -norm is not computable, but must be approximated.

Let us specify how to do this for specific encoder-decoder architectures

$$
\hat { \mathcal G } _ { \theta } = { \cal D } \circ g _ { \theta } \circ { \mathcal E } .\tag{5.3}
$$

We assume $\mathcal { E } : \mathcal { X } \to \mathbb { R } ^ { d _ { \mathcal { X } } } , \mathcal { D } : \mathbb { R } ^ { d _ { \mathcal { Y } } } \to \mathcal { Y }$ to be linear as defined in (3.3) via orthonormal bases $( \psi _ { \mathcal { X } , i } ) _ { i \in \mathbb { N } } , ~ ( \psi _ { \mathcal { Y } , j } ) _ { j \in \mathbb { N } }$ , of $\mathcal { X } , \mathcal { y }$ , i.e.

$$
\mathcal { E } ( x ) = ( \langle x , \psi _ { \mathcal { X } , i } \rangle _ { \mathcal { X } } ) _ { i = 1 } ^ { d _ { \mathcal { X } } } \in \mathbb { R } ^ { d _ { \mathcal { X } } } , \qquad \mathcal { D } ( ( c _ { j } ) _ { j = 1 } ^ { d _ { \mathcal { Y } } } ) = \sum _ { j = 1 } ^ { d _ { \mathcal { Y } } } c _ { j } \psi _ { \mathcal { Y } , j } .\tag{5.4}
$$

Due to Parseval’s identity it is natural to work directly on the coeficient sequences. We thus let

$$
\bar { \mathcal { R } } _ { m } ( \theta ) : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \| \underbrace { g _ { \theta } \circ \mathcal { E } ( x _ { j } ) - ( \langle y _ { j } , \psi _ { \mathcal { V } , i } \rangle ) _ { i = 1 } ^ { d _ { \mathcal { V } } } } _ { \in \mathbb { R } ^ { d _ { \mathcal { V } } } } \| _ { 2 } ^ { 2 } = \hat { \mathcal { R } } _ { m } ( \theta ) - \underbrace { \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \| ( \langle y _ { j } , \psi _ { \mathcal { V } , i } \rangle ) _ { i > d _ { \mathcal { V } } } \| _ { 2 } ^ { 2 } } _ { \mathrm { t r u n c a t i o n ~ e r r o r } } ,
$$

where the last equality follows by the fact that $( \psi _ { \mathcal { V } , i } ) _ { i \in \mathbb { N } }$ is a sequence of orthonormal vectors. Up to the additive and θ independent truncation error, this objective is thus equal to the empirical risk $\hat { \mathcal { R } } _ { m }$ and is easily computable. For simplicity we assumed orthonormal bases in Hilbert spaces; when working with p-Banach frames, the coeficient based losses additionally difer up to the frame constants.

## 5.2 Adding Derivatives

Let ${ \hat { \mathcal { G } } } _ { \theta } , \varepsilon$ and  be as in the previous subsection. As proposed and analyzed in [32, 116, 115, 101, 154], one can add gradient information to the objective, yielding so-called Sobolev training. For smooth classes of operators such as e.g. holomorphic maps , information on first derivatives of the operators to be emulated is often available. For example, in shape optimization, shape-gradients are in any case numerically evaluated (see, e.g., [115]). In practice, inclusion of diferential or gradient information in operator learning can significantly boost accuracy of the surrogate [115]. Derivative-informed operator learning requires training data of the form

$$
( x _ { j } , y _ { j } , Z _ { j } ) _ { j = 1 } ^ { m } \qquad \mathrm { w h e r e } \qquad y _ { j } = \mathcal { G } ( x _ { j } ) + \varepsilon _ { j } \in \mathcal { V } , \qquad Z _ { j } = D \mathcal { G } ( x _ { j } ) + E _ { j } \in L ( \mathcal { X } , \mathcal { Y } ) .
$$

As before $x _ { j } \in { \mathcal { X } } _ { : }$ , and $\varepsilon _ { j }$ as well as $E _ { j }$ are interpreted as noise or error terms.

Using again the square loss, the risk becomes

$$
\hat { \mathcal { R } } _ { m } ^ { 1 } ( \theta ) : = \hat { \mathcal { R } } _ { m } ( \theta ) + \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \lVert \underbrace { \mathcal { D } \circ D g _ { \theta } ( \mathcal { E } ( x _ { j } ) ) \circ \mathcal { E } ( x _ { j } ) } _ { = D \hat { \mathcal { G } } _ { \theta } ( x _ { j } ) } - Z _ { j } \rVert ^ { 2 } ,\tag{5.5}
$$

where $\| \cdot \|$ is a suitable norm (see below), $D g _ { \theta } \in \mathbb { R } ^ { d _ { \mathcal { V } } \times d _ { \mathcal { X } } }$ is the Jacobian of $g _ { \theta } : \mathbb { R } ^ { d _ { \mathcal { X } } }  \mathbb { R } ^ { d _ { \mathcal { Y } } }$ , and we used that $\mathcal { E } , \mathcal { D }$ are linear maps.

Since $D \hat { { \mathcal G } } _ { \theta } ( x _ { j } )$ and $Z _ { j }$ belong to $L ( \mathcal { X } , \mathcal { Y } )$ , it would be natural to take the operator norm in (5.5). In practice the Hilbert-Schmidt norm is preferred, as it simplifies the calculation. Recall that for a linear map $A : \mathcal { X }  \mathcal { Y }$ , its squared Hilbert Schmidt norm is $\begin{array} { r } { \| A \| _ { \mathrm { H S } ( \mathcal { X } , \mathcal { Y } ) } ^ { 2 } = \sum _ { i \in \mathbb { N } } \| A e _ { i } \| _ { \mathcal { Y } } ^ { 2 } = } \end{array}$ $\textstyle \sum _ { i , k \in \mathbb { N } } \left. A e _ { i } , f _ { k } \right. ^ { 2 }$ , where $( e _ { i } ) _ { i \in \mathbb { N } } , ( f _ { k } ) _ { k \in \mathbb { N } }$ , are (arbitrary) orthonormal bases of $\mathcal { X }$ and $\mathcal { V }$ . Choosing our orthonormal encoding and decoding bases $\psi _ { \mathcal { X } , i } , \psi _ { \mathcal { Y } , k }$ , we thus have

$$
\Vert A \Vert _ { \mathrm { H S } ( \mathcal { X } , \mathcal { Y } ) } ^ { 2 } = \underset { i \leq d _ { \mathcal { X } } , k \leq d _ { \mathcal { Y } } } { \sum } \langle A \psi _ { \mathcal { X } , i } , \psi _ { \mathcal { Y } , k } \rangle ^ { 2 } + \underset { \underset { \mathrm { t r u n c a t i o n ~ e r r o r } } { \sum } } { \sum } \langle A \psi _ { \mathcal { X } , i } , \psi _ { \mathcal { Y } , k } \rangle ^ { 2 } .
$$

Since $\left. D \hat { \mathcal { G } } _ { \theta } \psi _ { \mathcal { X } , i } , \psi _ { \mathcal { Y } , k } \right. = 0$ whenever either $i > d \chi$ or $k > d _ { \mathcal { Y } }$ (cf. (5.3) and (5.4)), we obtain (using the Hilbert-Schmidt norm in (5.5))

$$
\begin{array} { r } { \bar { \mathcal { R } } _ { m } ^ { 1 } ( \theta ) : = \bar { \mathcal { R } } _ { m } ( \theta ) + \frac { 1 } { m } \underset { j = 1 } { \overset { m } { \sum } } \lVert \underbrace { D g _ { \theta } ( \mathcal { E } ( x _ { j } ) ) - ( \langle Z _ { j } \psi _ { \mathcal { X } , i } , \psi _ { \mathcal { Y } , k } \rangle ) _ { i \leq d _ { \mathcal { X } } , k \leq d _ { \mathcal { Y } } } } _ { \in \mathbb { R } ^ { d _ { \mathcal { Y } } \times d _ { \mathcal { X } } } } \rVert _ { F } ^ { 2 } } \\ { = \hat { \mathcal { R } } _ { m } ^ { 1 } ( \theta ) - \underbrace { \frac { 1 } { m } \underset { j = 1 } { \overset { m } { \sum } } \Big ( \underset { i > d _ { \mathcal { Y } } } { \sum } \langle y _ { j } , \psi _ { \mathcal { Y } , i } \rangle ^ { 2 } + \underset { i > d _ { \mathcal { X } } } { \sum } \langle Z _ { j } \psi _ { \mathcal { X } , i } , \psi _ { \mathcal { Y } , k } \rangle ^ { 2 } \Big ) } _ { \mathrm { t r u r c a t i o n ~ e r r o r } } , } \end{array}\tag{5.6}
$$

where $\| \cdot \| _ { F }$ is the Frobenius norm of this matrix in $\mathbb { R } ^ { d _ { \mathcal { V } } \times d _ { \mathcal { X } } }$ . Since the truncation error in (5.6) is independent of $\theta ,$ minimizing $\mathcal { \bar { R } } _ { m } ^ { 1 } ( \theta )$ is equivalent to the minimization of $\hat { \mathcal { R } } _ { m } ^ { 1 } ( \theta )$ . The term $\mathcal { \bar { R } } _ { m } ^ { 1 } ( \theta )$ is easy to compute, and merely requires the Jacobians of $g _ { \theta }$ . In particular, all computations can be done at the level of coeficients.

## 6 Approximation Errors

For ease of presentation, throughout this section we focus once more on $\mathcal { X } , \mathcal { y }$ being separable Hilbert spaces and stable linear representations as described in Section 3.1. However, many arguments extend straightforwardly to the case of separable Banach spaces which admit a so-called Schauder basis (a countable basis with certain stability properties analogous to Example 3.3, 3rd bullet point). We refer to [143, 144] for constructions of such bases for a wide range of separable Sobolev and Besov spaces which arise as data and solution spaces for PDEs.

## 6.1 Universal Approximation

We first address the question of what kind of operators can be learned by finite-parametric surrogate families. For functions with finite dimensional in- and output, universal approximation with neural networks (and polynomials, wavelets etc.) is classical, e.g., [93]. In infinite dimensions, in the context of operator learning, the situation becomes a bit more subtle. The next statement highlights the importance of compactness of the input set.

Proposition 6.1. Let  be a Banach space. $L e t K \subseteq \mathcal { X }$ be bounded and such that $\bar { K }$ is not compact. Then there exists $\varepsilon > 0$ , such that for any $d \in \mathbb { N } _ { i }$ , any bounded linear $\mathcal { E } : \mathcal { X }  \mathbb { R } ^ { d }$ , and for any continuous map $\mathcal { F } : \mathbb { R } ^ { d }  \mathcal { X }$

$$
\operatorname* { s u p } _ { x \in K } \| x - \mathcal { F } \circ \mathcal { E } ( x ) \| _ { \mathcal { X } } \geq \varepsilon .
$$

Proof. Since $\bar { K }$ is closed but not compact, $\bar { K }$ is not totally bounded, meaning there exists $\varepsilon > 0$ such that $\begin{array} { r } { \bigcup _ { x \in K } B _ { 2 \varepsilon } ( x ) \supseteq \bar { K } } \end{array}$ has no finite subcover. From this we can conclude that there exists a sequence $( x _ { j } ) _ { j \in \mathbb { N } } \subseteq K$ such that

$$
\| x _ { i } - x _ { j } \| _ { X } \geq 2 \varepsilon \qquad \forall i \neq j .\tag{6.1}
$$

On the other hand $\mathcal { E } ( x _ { j } ) _ { j \in \mathbb { N } }$ is a bounded subset of $\mathbb { R } ^ { d }$ , and therefore has a subsequence that is a Cauchy sequence. Without loss of generality, let this sequence be $\mathcal { E } ( x _ { j } ) _ { j \in \mathbb { N } }$ . Due to the continuity of $\mathcal { F }$

$$
\operatorname* { l i m } _ { N  \infty } \operatorname* { s u p } _ { i , j \geq N } \| \mathcal { F } ( \mathcal { E } ( x _ { i } ) ) - \mathcal { F } ( \mathcal { E } ( x _ { j } ) ) \| _ { \mathcal { X } } = 0 .
$$

Together with (6.1) we conclude that

$$
2 \operatorname* { s u p } _ { i \in \mathbb { N } } \| x _ { i } - \mathcal { F } ( \mathcal { E } ( x _ { i } ) ) \| _ { \mathcal { X } } \ge \operatorname* { s u p } _ { i \ne j \in \mathbb { N } } | \| x _ { i } - x _ { j } \| _ { \mathcal { X } } - \| \mathcal { F } ( \mathcal { E } ( x _ { i } ) ) - \mathcal { F } ( \mathcal { E } ( x _ { j } ) ) \| _ { \mathcal { X } } | \ge 2 \varepsilon .
$$

The proposition states that if $\bar { K }$ is not compact, and we use a bounded linear encoder $\mathcal { E }$ then no architecture that processes the encoded input $\mathcal { E } ( x )$ continuously (which covers all common architectures) is capable of uniformly approximating the identity operator $\mathcal { G } = \mathrm { I d }$ on K. The statement does not extend to nonlinear encoders $\mathcal { E } .$ . However, in most practical situations $\mathcal { X }$ is a function space continuously embedded into the continuous functions, and the first step is to evaluate the input on a grid in the physical (space-time) domain; this corresponds to a linear encoding. For any such method, uniform approximation of operators on non-compact input sets in $\mathcal { X }$ is therefore in general impossible. We emphasize that this limitation does not apply to approximation with respect to weaker norms, such as $L ^ { p }$ norms, see, e.g., [87, 71].

A similar argument implies that for linear decoders, the operators’ response manifolds $\mathcal { G } ( x ) _ { x \in K } \subseteq$ must also be relatively compact; again this is not required for nonlinear decoders. We do not discuss this in further detail and instead turn to a positive result on universal approximation in Hilbert spaces from [133]. To state it we first introduce the set of admissible activation functions from [93]

$$
\begin{array} { r l } & { \mathcal { M } : = \{ \sigma \in L _ { \mathrm { l o c } } ^ { \infty } ( \mathbb { R } ) \mathrm { ~ i s ~ n o t ~ a ~ p o l y n o m i a l , ~ a n d ~ t h e ~ c l o s u r e ~ o f ~ } } \\ & { \qquad \mathrm { ~ t h e ~ p o i n t s ~ o f ~ d i s c o n t i n u i t y ~ f o r m ~ a ~ L e b e s g u e ~ n u l l ~ s e t } \} , } \end{array}
$$

where $L _ { \mathrm { l o c } } ^ { \infty } ( \mathbb { R } )$ is the set of (possibly unbounded) measurable functions on R which are (essentially) bounded on each bounded subinterval of R. We consider surrogates of the form

$$
\begin{array} { r l } & { \{ \mathcal { D } \circ g \circ \mathcal { E } : d _ { \mathcal { X } } , d _ { \mathcal { Y } } \in \mathbb { N } ; \mathcal { E } : \mathcal { X } \to { \mathbb { R } } ^ { d _ { \mathcal { X } } } , \ \mathcal { D } : { \mathbb { R } } ^ { d _ { \mathcal { Y } } } \to \mathcal { Y } \ \mathrm { a r e ~ l i n e a r ~ e n c o d e r s } / \operatorname* { d e c o d e r s } } \\ & { \qquad \mathrm { b a s e d ~ o n ~ s t a b l e ~ r e p r e s e n t a t i o n s ~ a s ~ i n ~ S e c t i o n ~ 3 . 1 } ; } \\ & { \qquad \quad g : { \mathbb { R } } ^ { d _ { \mathcal { X } } } \to { \mathbb { R } } ^ { d _ { \mathcal { Y } } } \mathrm { ~ i s ~ a ~ f e e d f o r w a r d ~ N N ~ w i t h ~ a c t i v a t i o n ~ f u n c t i o n ~ } \sigma \} . } \end{array}\tag{6.2}
$$

The next theorem corresponds to [133, Theorem 3.1].

Theorem 6.2. Let $x , y$ be separable Hilbert spaces. Then for every continuous $\mathcal { G } : \mathcal { X }  \mathcal { Y }$ there exists a sequence $\mathcal { G } ^ { k } = \mathcal { D } ^ { k } \circ g ^ { k } \circ \mathcal { E } ^ { k } , k \in \mathbb { N }$ , with architecture (6.2) such that

$$
\forall x \in \mathcal { X } : \qquad \operatorname* { l i m } _ { k \to \infty } \mathcal { G } ^ { k } ( x ) = \mathcal { G } ( x ) .
$$

The convergence is uniform on every compact subset K of .

Remark 6.3. Given a class of approximating mappings $\mathcal { H } _ { \mathrm { ~ } }$ , universal approximation theorems are often stated in the following form:

For every $\epsilon > 0$ and every compact $K \subseteq { \mathcal { X } }$ there exists $\hat { \mathcal { G } } _ { \epsilon , K } \in \mathcal { H }$ such

that s $\begin{array} { r } { \operatorname* { u p } _ { x \in K } \lVert \mathcal { G } ( x ) - \hat { \mathcal { G } } _ { \epsilon , K } ( x ) \rVert _ { \mathcal { X } } < \epsilon . } \end{array}$

If is finite dimensional, this is equivalent to Theorem 6.2 as is easy to see $( e . g . , \ I 1 2 3 ,$ ${ P r o p . ~ 3 . 4 J } )$ . In the infinite dimensional case, Theorem 6.2 gives a strictly stronger statement.

Remark 6.4. A typical setting where compactness holds is the following: let $\mathbf { D } \subset \mathbb { R } ^ { n }$ be a bounded domain, with ∂D Lipschitz. Then for $s > s ^ { \prime } \ge 0$ , and any $1 \leq p , q \leq \infty$ , it holds $B _ { p q } ^ { s } ( \mathrm { D } ) \subset B _ { p q } ^ { s ^ { \prime } } ( \mathrm { D } )$ with compact embedding (Rellich’s theorem) $\boldsymbol { \left. \mu \mathcal { 3 } \right. }$

Universal approximation theorems for other architectures can for instance be found in [87, 78, 57, 82]. For the recently popular transformer architectures, universal approximation has been shown e.g. in [17].

## 6.2 $C ^ { k }$ Operators

After having established universal approximation, we next discuss convergence rates over classes of operators. Classical approximation theory is concerned with smoothness spaces, such as the k times diferentiable functions. This motivates us to consider the unit ball of the $C ^ { k }$ operators from $\mathcal { X }  \mathcal { V }$ , i.e.

$$
C _ { 1 } ^ { k } ( \mathcal { X } , \mathcal { Y } ) : = \{ \mathcal { G } : \mathcal { X } \to \mathcal { Y } : \operatorname* { m a x } _ { 0 \leq j \leq k } \operatorname* { s u p } _ { x \in \mathcal { X } } \| D ^ { j } \mathcal { G } ( x ) \| \leq 1 \} .
$$

As we have seen, compactness of the input set is crucial, and therefore we focus on the uniform approximation of these operators on some compact set $K \subseteq { \mathcal { X } }$ . In order to qualify as true “operator learning”, K should not be contained in a finite dimensional subspace or manifold; this would render the problem efectively finite dimensional. To formalize this, we assume that K contains a full ball within a compactly embedded subspace:

Assumption 6.5. Let $\tilde { \mathcal X }$ be an infinite dimensional Hilbert space that embeds compactly in  . There exists $r > 0$ such that $\{ x \in \tilde { \mathcal { X } } : \| x \| _ { \tilde { \mathcal { X } } } \leq r \} \subseteq K$

We start by recalling a by now classical result regarding continuous nonlinear N-widths of unit balls in $C ^ { k } ( [ 0 , 1 ] ^ { d } )$ from [124].

Theorem 6.6. Let d, $k \in \mathbb N$ . Then there exists $C > 0$ such that for every map $\bar { \mathcal { A } } : \mathbb { R } ^ { N } \to C ^ { 0 } ( [ 0 , 1 ] ^ { d } )$ and every continuous map $\bar { \Theta } : C ^ { 0 } ( [ 0 , 1 ] ^ { d } ) \to \mathbb { R } ^ { N }$ there exists $g \in C ^ { k } ( [ 0 , 1 ] ^ { d } )$ with $\| g \| _ { C ^ { k } ( [ 0 , 1 ] ^ { d } ) } \leq 1$ such that

$$
\| g - \bar { \mathcal { A } } ( \bar { \Theta } ( g ) ) \| _ { C ^ { 0 } ( [ 0 , 1 ] ^ { d } ) } \geq C N ^ { - k / d } \qquad \forall N \in \mathbb { N } .\tag{6.3}
$$

An interpretation of the theorem is as follows: the mapping $\bar { A }$ takes N parameters to a function in $C ^ { 0 } ( [ 0 , 1 ] ^ { d } )$ . We can think of it as some neural network (or other) framework parametrizing a function class with a finite number of parameters $\boldsymbol { \theta } \in \mathbb { R } ^ { N }$ . The mapping Θ<sup>¯</sup> takes a $g \in C ^ { 0 } ( [ 0 , 1 ] ^ { d } )$ and returns a corresponding parameter vector $\bar { \Theta } ( g ) \in \mathbb { R } ^ { N }$ . The composition $\bar { A } \circ \bar { \Theta }$ should then yield an approximation of the identity. It thus gives a lower bound on how well the function class $\{ g \in C ^ { k } ( [ 0 , 1 ] ^ { d } ) : \| g \| _ { C ^ { k } } \leq 1 \}$ can be uniformly approximated with N parameters. Importantly, the parameter assignment Θ<sup>¯</sup> is assumed continuous, i.e. functions that are close have similar parameters. This is a form of stability, that is reasonable to request for a practical algorithm. Without such a restriction, the quantity (6.3) becomes zero, due to the existence of space-filling curves [124].

Remark 6.7. Typical neural network training does not satisfy continuity of the parameter assignment Θ<sup>¯</sup> : Random initialization and the use of stochastic optimization algorithms produce diferent weights on diferent runs, even for the same data. Nevertheless, some form of stability is necessary for any practical algorithm to avoid the pathology of space-filling curves. As such, while a simplification, the continuity hypothesis serves as a mathematical idealization, and the lower bounds remain meaningful to indicate the limits of stable approximation methods.

Proposition 6.8. Let $x , y$ be two separable Hilbert spaces. Let $K \subseteq \mathcal X$ be compact and satisfy Assumption 6.5. Then for any $\alpha > 0$ there exists $C _ { \alpha } > 0$ such that

$$
\operatorname* { i n f } _ { \substack { \Theta : C ^ { 0 } ( \mathcal { X } , \mathcal { Y } )  \mathbb { R } ^ { N } \ c o n t . \quad \mathcal { G } \in C _ { 1 } ^ { k } ( \mathcal { X } , \mathcal { Y } ) } } \operatorname* { s u p } _ { x \in K } \Vert \mathcal { G } ( x ) - \mathcal { A } ( \Theta ( \mathcal { G } ) ) ( x ) \Vert _ { \mathcal { Y } } \geq C _ { \alpha } N ^ { - \alpha } \qquad \forall N \in \mathbb { N } .
$$

Proof. It sufices to consider the case $\mathcal { V } = \mathbb { R } \colon \mathrm { H } \ \mathcal { V }$ is any other Hilbert space, then the statement follows by lower bounding the supremum over $C _ { 1 } ^ { k } ( \mathcal { X } , \mathrm { s p a n } ( e ) ) \ \subseteq \ C _ { 1 } ^ { k } ( \mathcal { X } , \mathcal { Y } )$ , where $e \in \mathcal { V }$ is an arbitrary norm one vector; the space $C _ { 1 } ^ { k } ( \mathcal { X } , \operatorname { s p a n } ( \bar { e } ) )$ can be identified with $C _ { 1 } ^ { k } ( \mathcal { X } , \mathbb { R } )$

Step 1. Fix $d \in \mathbb { N }$ so large that $k / d < \alpha$ . Let $( \tilde { e } _ { i } ) _ { i \in \mathbb { N } }$ be an ONB of $\tilde { \mathcal X }$ , with <sup>˜</sup> as in Assumption 6.5. By Gram-Schmidt orthogonalization, we can create an orthonormal system $( e _ { i } ) _ { i \in \mathbb { N } }$ in from this sequence. By Assumption 6.5

$$
\left\{ \sum _ { i = 1 } ^ { d } y _ { i } { \tilde { e } } _ { i } : \sum _ { i = 1 } ^ { d } y _ { i } ^ { 2 } \leq r \right\} \subseteq K .
$$

Hence, for some fixed $\lambda > 0$ suficiently small,

$$
\left\{ \lambda \sum _ { i = 1 } ^ { d } y _ { i } e _ { i } : 0 \leq y _ { i } \leq 1 \right\} \subseteq K .
$$

Step 2. Now let $\mathcal { A } : \mathbb { R } ^ { N } \to C ^ { 0 } ( K , \mathbb { R } )$ be arbitrary, and let $\Theta : C ^ { 0 } ( \mathcal { X } , \mathbb { R } ) \to \mathbb { R } ^ { N }$ be continuous. Set

$$
\bar { \mathcal { A } } : = \left\{ \begin{array} { l l } { \mathbb { R } ^ { N } \to C ^ { 0 } ( [ 0 , 1 ] ^ { d } ) } \\ { \theta \mapsto \Big ( ( y _ { 1 } , \dots , y _ { d } ) \mapsto \mathcal { A } ( \theta ) ( \lambda \sum _ { i = 1 } ^ { d } y _ { i } e _ { i } ) \Big ) . } \end{array} \right.
$$

Next let $E : C ^ { 0 } ( [ 0 , 1 ] ^ { d } ) \to C ^ { 0 } ( \mathbb { R } ^ { d } )$ be a linear extension operator satisfying for all $f \in C ^ { 0 } ( [ 0 , 1 ] ^ { d } )$ and $g \in C ^ { k } ( [ 0 , 1 ] ^ { d } )$

$$
\| E ( f ) \| _ { C ^ { 0 } ( \mathbb { R } ^ { d } ) } \leq C _ { E } \| E ( f ) \| _ { C ^ { 0 } ( [ 0 , 1 ] ^ { d } ) } \qquad \mathrm { ~ a n d ~ } \qquad \| E ( g ) \| _ { C ^ { k } ( \mathbb { R } ^ { d } ) } \leq C _ { E } \| E ( g ) \| _ { C ^ { k } ( [ 0 , 1 ] ^ { d } ) } ,
$$

for some fixed constant $C _ { E } > 0$ . Existence of such (Whitney) extension operators is well-known (e.g. [138, Chap. VI]). For arbitrary $f \in C ^ { j } ( [ 0 , 1 ] ^ { d } ) , j \in \{ 0 , k \}$ ，

$$
\mathcal { G } _ { g } ( \boldsymbol { x } ) : = E ( g ) ( \lambda ^ { - 1 } \left. x , e _ { 1 } \right. , \ldots , \lambda ^ { - 1 } \left. x , e _ { d } \right. )
$$

defines an operator in $C ^ { j } ( \mathcal { X } , \mathbb { R } )$ . Moreover, there exists a constant $C _ { \mathcal { G } }$ depending on λ and $C _ { E }$ such that for $j \in \{ 0 , k \}$

$$
\begin{array} { r } { \| \mathcal { G } _ { g } \| _ { C ^ { j } ( \mathcal { X } , \mathbb { R } ) } \leq C _ { \mathcal { G } } \| g \| _ { C ^ { j } ( [ 0 , 1 ] ^ { d } ) } . } \end{array}
$$

With this we define

$$
\bar { \Theta } : = \left\{ \begin{array} { l l } { C ^ { 0 } ( [ 0 , 1 ] ^ { d } ) \to \mathbb { R } ^ { N } } \\ { g \mapsto \Theta ( \mathcal { G } _ { g } ) . } \end{array} \right.
$$

Note that continuity of Θ implies continuity of Θ<sup>¯</sup> .

Step 3. The definitions in Step 2 imply for all $g \in C ^ { k } ( [ 0 , 1 ] ^ { d } )$

$$
\bar { \mathcal { A } } ( \bar { \Theta } ( g ) ) ( \pmb { y } ) = \mathcal { A } ( \Theta ( \mathcal { G } _ { g } ) ) \Big ( \lambda \sum _ { i = 1 } ^ { d } y _ { i } e _ { i } \Big ) \qquad \forall \pmb { y } \in [ 0 , 1 ] ^ { d } .
$$

Therefore

$$
\begin{array} { r l } { \underset { \mathcal { G } \in { \cal C } _ { 1 } ^ { k } } { \operatorname* { s u p } } | \mathcal { G } ( x ) - \mathcal { A } ( \Theta ( \mathcal { G } ) ) ( x ) | \geq \underset { g \in { \cal C } ^ { k } ( [ 0 , 1 ] ^ { d } ) } { \operatorname* { s u p } } } & { \underset { x \in K } { \operatorname* { s u p } } | \mathcal { G } _ { g } ( x ) - \mathcal { A } ( \Theta ( \mathcal { G } _ { g } ) ) ( x ) | } \\ & { \quad \quad \quad \quad \quad \| g _ { | _ { { \cal C } ^ { k } } \times { \cal C } _ { 0 } ^ { - 1 } } } \\ & { \quad \quad \quad \quad \quad \geq \underset { g \in { \cal C } ^ { k } ( [ 0 , 1 ] ^ { d } ) } { \operatorname* { s u p } } } & { \underset { x \in \{ \lambda \sum _ { i = 1 } ^ { d } y _ { i } e _ { i } : 0 \leq y _ { i } \leq 1 \} } { \operatorname* { s u p } } | \mathcal { G } _ { g } ( x ) - \mathcal { A } ( \Theta ( \mathcal { G } _ { g } ) ) ( x ) | } \\ & { \quad \quad \quad \quad \quad \| g _ { | _ { { \cal C } ^ { k } } \times { \cal C } _ { 0 } ^ { - 1 } } } \\ & { = \underset { g \in { \cal C } ^ { k } ( [ 0 , 1 ] ^ { d } ) } { \operatorname* { s u p } } } & { \underset { y \in [ 0 , 1 ] ^ { d } } { \operatorname* { s u p } } | g ( y ) - \bar { \mathcal { A } } ( \bar { \Theta } ( g ) ) ( x ) | . } \\ & { \quad \quad \quad \quad \| g _ { | _ { { \cal C } ^ { k } } \times { \cal C } _ { 0 } ^ { - 1 } } } \end{array}
$$

By Theorem 6.6, the last quantity is lower bounded by $C N ^ { - k / d }$ . Since and Θ were arbitrary, and $\alpha \leq k / d ,$ , this concludes the proof. □

According to the above theorem, under the assumption of continuous parameter dependence, there cannot exist an architecture capable of uniformly approximating $C ^ { k }$ operators at an algebraic rate. This is a fundamental restriction related to this function class. We can draw a similar conclusion for the amount of data that is required to learn $C ^ { k }$ operators.

Proposition 6.9. Let  ,  be two separable Hilbert spaces. Let $K \subseteq \mathcal X$ be compact and satisfy Assumption 6.5, and let $x _ { 1 } , \dotsc , x _ { N } \in K$ . Then for every $\alpha > 0$ there exists $C _ { \alpha } > 0$ such that

$$
\operatorname* { i n f } _ { \substack { \mathcal { A } \colon \mathcal { Y } ^ { 0 } ( K , \mathcal { Y } ) } } \operatorname* { s u p } _ { \mathcal { G } \in C _ { 1 } ^ { k } ( \mathcal { X } , \mathcal { Y } ) } \operatorname* { s u p } _ { x \in K } \| \mathcal { G } ( x ) - \mathcal { A } ( \mathcal { G } ( x _ { 1 } ) , \ldots , \mathcal { G } ( x _ { N } ) ) ( x ) \| _ { \mathcal { Y } } \geq C _ { \alpha } N ^ { - \alpha } \qquad \forall N \in \mathbb { N } .
$$

Proposition 6.9 states that the amount of data N required to reliably reconstruct an operator $\mathcal { G } \in C ^ { k }$ needs to increase faster than any polynomial; also see [80] for more precise statements and sharper (exponential) bounds. For $\mathcal { V } = \mathbb { R }$ , the statement is simply a special case of Proposition 6.8 with $\Theta ( { \mathcal { G } } ) = { \mathcal { G } } ( x _ { 1 } ) , \dots , { \mathcal { G } } ( x _ { N } )$ . The case of general $\mathcal { V }$ can again be reduced to the real-valued one, similar as in Proposition 6.8.

Related results on parametric and data complexity can found in [107, 106, 80, 85]. E.g., in [85], rather than assuming continuous parameter dependence, the perspective of quantization is taken. Specifically, the metric entropy is used to derive lower bounds on the number of bits required to approximate the class of Lipschitz operators. Moreover, [133] provides a construction which does achieve algebraic rates without the CoD for the approximation of Lipschitz operators. The proof therefore relies on discontinuous parameter dependence, however.

## 6.3 Holomorphic Operators

We have argued that in general a form of compactness of the input set is required, and that classical function spaces of finite smoothness are too big to be amenable for operator learning (with uniform convergence). Thus, further restrictions are required. Motivated by well-known properties of parametric PDEs, a common class to consider are holomorphic operators [29, 27, 71, 3]. In this section we discuss a convergence result from [71].

## 6.3.1 Smoothness Spaces

We introduce a scale of smoothness spaces. To this end we assume linear encoders and decoders defined over Hilbert frames $( \psi _ { \mathcal { X } , j } , \varphi _ { \mathcal { X } , j } ) _ { j \in \mathbb { N } }$ and $( \psi _ { \mathcal { V } , j } , \varphi y , _ { j } ) _ { j \in \mathbb { N } }$ as introduced Section 3.1. For $s ,$ $t \geq 0$ we let

$$
\mathcal { X } ^ { s } : = \left\{ x \in \mathcal { X } : \sum _ { j \in \mathbb { N } } \varphi x , j ( x ) ^ { 2 } j ^ { 2 s } < \infty \right\} , \qquad \mathcal { Y } ^ { t } : = \left\{ y \in \mathcal { Y } : \sum _ { j \in \mathbb { N } } \varphi y , j ( y ) ^ { 2 } j ^ { 2 t } < \infty \right\} ,\tag{6.4}
$$

with the norms in those spaces given by the respective weighted sums. In the following we consider corresponding encoder, decoder maps

$$
\mathcal { E } : \mathcal { X } \to \mathbb { R } ^ { d _ { \mathcal { X } } } , \qquad \mathcal { D } : \mathbb { R } ^ { d _ { \mathcal { Y } } } \to \mathcal { Y }\tag{6.5}
$$

as in (3.3) with the above frames.

Example 6.10. Let $\psi _ { j } ( z ) ~ = ~ \sqrt { 2 } \sin ( \pi j z ) ~ f o r ~ z ~ \in ~ [ 0 , 1 ]$ and $j \in \mathbb N$ . Then $( \psi _ { j } ( z ) ) _ { j \in \mathbb { N } }$ together with $\begin{array} { r } { \varphi _ { j } ( f ) = \int _ { 0 } ^ { 1 } f ( z ) \psi _ { j } ( z ) \mathrm { d } } \end{array}$ z is $a \ ( t i g h t )$ Hilbert frame<sup>7</sup> on $\mathcal { X } = L ^ { 2 } ( [ 0 , 1 ] )$ . Moreover, $\mathcal { X } ^ { s } = \{ f \in$ $H ^ { s } ( [ 0 , 1 ] ) : f ( \bar { 0 ) } = f ( 1 ) = 0 \}$

As stated in [71, Lemma 2], for $0 \leq s ^ { \prime } < s$ , the embedding $\mathcal X ^ { s } \hookrightarrow \mathcal X ^ { s ^ { \prime } }$ is compact. In particular, balls in $\mathcal { X } ^ { s }$ are compact in $x ;$ they thus satisfy the minimal criteria of input sets required for the uniform approximation of an operator $\mathcal { G } : \mathcal { X }  \mathcal { Y }$ , see Section 6.1.

## 6.3.2 Convergence Rates

To state the convergence rate result, we first make the holomorphy assumption on $\mathcal { G }$ more precise. To this end we denote by $\mathcal { X } _ { \mathbb { C } } ^ { s }$ the complexification of the Hilbert space $\mathcal { X } ^ { s }$ , and similar for $\mathcal { V }$

Assumption 6.11. Let $r > 0$ . There exists $s > 1 , t > 0 , M < \infty$ , and an open set $O _ { \mathbb { C } } \subseteq { \mathcal { X } } _ { \mathbb { C } }$ containing $\begin{array} { r } { \{ r \sum _ { j \in \mathbb { N } } j ^ { - s } \bar { x } _ { j } \psi _ { \mathcal { X } , j } : \bar { \pmb { x } } \in [ - 1 , 1 ] ^ { \mathbb { N } } \} } \end{array}$ , such that $\begin{array} { r } { \operatorname* { s u p } _ { x \in O _ { \mathbb { C } } } \| \mathcal G ( x ) \| _ { \mathcal { V } _ { \mathbb { C } } ^ { t } } \leq M } \end{array}$ and $\mathcal { G } : O _ { \mathbb { C } } \to \mathcal { V } _ { \mathbb { C } }$ is holomorphic.

The assumption states two things: First, $\mathcal { G }$ allows a complex diferentiable extension to an open subset $O _ { \mathbb { C } }$ of $\mathcal { X } _ { \mathbb { C } }$ . Second, its outputs belong to $\mathcal { V } _ { \mathbb { C } } ^ { t }$ . The first part of the assumption implies that the operator response depends very smoothly on the input. In particular it is not just k times diferentiable, but it is even analytic. The set $O _ { \mathbb { C } }$ of analyticity describes a region in the complexified space related to the ball of radius r in $\mathcal { X } ^ { s }$ centered at 0. The assumption essentially requires that is not just well-behaved on the real-valued ball, but also in a suficiently large complex neighborhood around it. This is a common requirement for deriving approximation results. The second part of the assumption implies a certain decay of the coeficients of the output functions. While the assumption looks rather technical, the next example demonstrates its wide applicability in the context of learning PDE solutions.

Example 6.12. $L e t  { \mathrm { D } } \subseteq \mathbb { R } ^ { d }$ be a bounded Lipschitz domain, and denote by $\mathscr { X } = L ^ { \infty } ( \mathrm { D } )$ the Lebesgue space of essentially bounded functions $x : \mathrm { D }  \mathbb { R }$ , and by $\mathcal { V } = H _ { 0 } ^ { 1 } ( \mathrm { D } )$ the Sobolev space of square integrable functions $y : \mathrm { D }  \mathbb { R }$ , whose first weak derivative is also square integrable.

Additionally fix $f \in \mathcal { V } ^ { \prime }$ and denote

$$
\mathcal { N } ( x , y ) : = \nabla \cdot ( x \nabla y ) - f \in \mathcal { y } ^ { \prime } .\tag{6.6}
$$

For any $x _ { 0 } \in L _ { a _ { - } } ^ { \infty } ( \mathrm { D } ) \subset \mathcal { X } \ ( c p . \ ( 2 . 1 ) )$ with strictly positive essential infimum on D, the Lax-Milgram Lemma guarantees that $\mathcal { N } ( x _ { 0 } , \cdot ) : \mathcal { V } \to \mathcal { Y } ^ { \prime }$ is a bounded isomorphism. In particular, there exists a unique $y _ { 0 } \in \mathcal { V }$ such that $\mathcal { N } ( x _ { 0 } , y _ { 0 } ) = 0$ , and the Fréchet derivative $D _ { y } \mathcal { N } ( x _ { 0 } , y _ { 0 } ) ( v ) = \nabla { \cdot } ( x _ { 0 } \nabla v ) \in \mathcal { Y } ^ { \prime }$ $v \in \mathcal { V }$ , is an isomorphism from $\mathcal { V }  \mathcal { V } ^ { \prime }$ . The implicit function theorem in Banach spaces, $e . g .$ $I 1 5 7 ,$ Theorem $4 . B J ,$ then implies the existence of a neighbourhood $O \subseteq { \mathcal { X } }$ of $x _ { 0 }$ and a diferentiable function $Y : O  \mathcal { V }$ such that $\mathcal { N } ( x , Y ( x ) ) = 0$ for all $x \in O$ . Note that Y is the mapping taking a difusion coeficient x to the solution y of the Darcy flow $P D E \left( 6 . 6 \right)$ . This argument extends directly to the complexified spaces, and can thus be used to verify that the solution map is locally complex diferentiable in a neighbourhood of $x _ { 0 }$ in $\mathcal { X } _ { \mathbb { C } }$ . Together with the compactness of the parameter set, one can then derive holomorphy and uniform boundedness as stated in the assumption; for more details also see the concrete example in Section 2.1, and, $e . g . , \ [ 1 5 6 ,$ , Sec. 1.2] for the general treatment. This corresponds to the first part of Assumption 6.11.

The second part of the assumption states that the solution map Y maps into a space $\mathcal { V } _ { \mathbb { C } } ^ { t }$ , where the parameter $t > 0$ quantifies the coeficient decay. For PDE models like (6.6), this can be understood as a regularity assumption on the PDE solution. For example, i $f t \geq 0 , f \in H ^ { t } ( \mathrm { D } ) , x \in C ^ { t + 1 } ( \mathrm { D } )$ , and D is a C<sup>t+2</sup>-domain, then classical regularity theory implies $Y ( x ) \in H ^ { t + 2 } ( \mathrm { D } ) , \ e . g . , \ / 4 ^ { g } ,$ Sec. $6 . 3 J .$ In suitable representation systems, this translates to coeficient decay of the type (6.4), cf. Example 3.3.

We emphasize that the above argument relies only on two crucial observations: the diferential operator  depends holomorphically (in the present case even linearly) on the parameter x, and $D _ { y } \mathcal { N } ( x _ { 0 } , y _ { 0 } ) : \mathcal { V } _ { \mathbb { C } }  \mathcal { V } _ { \mathbb { C } } ^ { \prime }$ is an isomorphism; the latter usually follows by well-posedness of the PDE. For this reason the same argument extends to a large range of PDEs and related problems (under additional suitable assumptions), $e . g .$ , Maxwell’s equations $I 7 4 J ,$ the Navier-Stokes equations [30], or Boundary integral operators $I 6 8 J ;$ also see [28] for a general treatment.

This will allow to approximate the output within finite dimensional spaces. There holds the next theorem [71, Corollary 1] (see the discussion above Theorem 2.7 for the notion of network size).

Then there exists a constant $C > 0$ such that for all $N \in  { \mathbb { N } }$ there exist $d _ { \mathcal { X } } , d _ { \mathcal { Y } } \in \mathbb { N }$ in (6.5), and a ReLU neural network $g ^ { N } : \mathbb { R } ^ { d _ { \mathcal { X } } }  \mathbb { R } ^ { d _ { \mathcal { Y } } }$ of size $O ( N )$ such that

$$
\operatorname* { s u p } _ { x \in B _ { r } ( \mathcal { X } ^ { s } ) } \| \mathcal { G } ( x ) - \mathcal { D } \circ g ^ { N } \circ \mathcal { E } ( x ) \| _ { \mathcal { V } } \leq C N ^ { - \operatorname* { m i n } \{ s - 1 , t \} + \delta } .
$$

Here $B _ { r } ( \mathcal { X } ^ { s } )$ denotes the ball of radius r and center $0 \in \mathcal { X } ^ { s }$

Remark 6.14. In the cited reference, the approximator network $g ^ { N }$ has been constructed with ReLU activations. However, we expect that the same results hold also with other, smooth activations such as tanh() or sigmoidal activations.

Remark 6.15. Sparse-grid interpolation surrogates can achieve the same rate in terms of the amount of data $m _ { ; }$ , $/ 7 1$ , Theorem $\it 4 J .$ As mentioned, these surrogates are explicitly computable without the need $f o r$ training, under the provision that one can numerically access $( { } ^ { \circ } \mathrm { { \ ' q u e r y } } ^ { \prime \prime } ) ~ \mathcal { G }$ with any admissible input data, to prescribed accuracy or “fidelity”. In many practical situations, such as data-assimilation, this is not possible and one has to rely on a probabilistic setting, assuming training data is drawn i.i.d. subject to a probability measure charging the ensemble of admissible inputs. The assumption that this measure be known is quite strong and not easily verifiable.

Theorem 6.13 states that it is possible to approximate holomorphic operators on balls in $\mathcal { X } ^ { s }$ While such a ball is not compact in $\mathcal { X } ^ { s }$ , it is compact in the topology of $\mathcal { X } .$ Therefore, Theorem 6.13 does not contradict Proposition 6.1 if we take $\mathcal { G } = \mathrm { I d }$ and $\mathcal { V } = \mathcal { X }$ . Sharper results, convergence in $L ^ { 2 }$ , and further details can be found in [71]. The proof of the theorem is intricate and long, so we do not repeat the full argument. We do give some intuition however.

It is instructive to consider the Fourier type frame basis in Example 6.10 on $\mathcal { X } = \mathcal { Y } = L ^ { 2 } ( [ 0 , 1 ] )$ Suppose that $\mathcal G : \mathcal X \to \mathcal y ^ { t }$ . The encoder $\mathcal { E }$ computes a finite number of input coeficients, d. The neural network $g ^ { N } : \mathbb { R } ^ { d _ { \mathcal { X } } }  \mathbb { R } ^ { d _ { \mathcal { Y } } }$ then approximates the action of the operator in the coeficient space; it learns the mapping for the first $d _ { \mathcal { Y } }$ output coeficients $( g _ { i } ) _ { i = 1 } ^ { d y }$ . Finally, the decoder $\mathcal { D }$ synthesizes these $d y$ coeficients back into a function in $\mathcal { V } = L ^ { 2 } ( [ 0 , 1 ] )$

In the current setting, for $x \in \mathcal { X } = L ^ { 2 } ( [ 0 , 1 ] )$ with

$$
\bar { \pmb { x } } = ( \bar { x } _ { i } ) _ { i \in \mathbb { N } } , \qquad \bar { x } _ { i } = \langle \pmb { x } , \sin ( i \pi \cdot ) \rangle _ { L ^ { 2 } ( [ 0 , 1 ] ) }
$$

we want to approximate the operator $\mathcal { G }$ on the set

$$
B _ { r } ( \mathcal { X } ^ { s } ) = \left\{ x \in \mathcal { X } : \sum _ { j \in \mathbb { N } } \bar { x } _ { j } ^ { 2 } j ^ { 2 s } < r ^ { 2 } \right\} \subseteq H ^ { s } ( [ 0 , 1 ] ) .
$$

With the above choice of basis,

$$
g _ { i } ( \bar { \pmb x } ) = \left. \sin ( i \pi \cdot ) , \mathcal G \Big ( \sum _ { j \in \mathbb { N } } \bar { x } _ { j } \sin ( \pi j \cdot ) \Big ) \right. _ { L ^ { 2 } ( [ 0 , 1 ] ) } .
$$

In words, $g _ { i }$ maps the sequence of Fourier coeficients of the input $x \in \mathcal { X }$ , to the ith Fourier coeficient of the output $\mathcal G ( x ) \in \mathcal { V }$

Due to the holomorphy of $\mathcal { G } _ { ; }$ , the map $g _ { i }$ is holomorphic in each $\bar { x } _ { i }$ . A particular consequence is that $g _ { i }$ allows (locally) a Taylor polynomial chaos expansion

$$
g _ { i } ( \bar { \pmb x } ) = \sum _ { \pmb { \nu } \in \mathscr { F } } \underbrace { \frac { \partial ^ { \nu } g _ { i } ( \mathbf { 0 } ) } { \nu ! } } _ { = : c _ { i , \nu } } \bar { \pmb x } ^ { \nu } \simeq \sum _ { \pmb { \nu } \in \Lambda _ { i } } c _ { i , \nu } \bar { \pmb x } ^ { \nu } \qquad \forall i \in \mathbb { N } ,\tag{6.7}
$$

whereas before $\Lambda _ { i } \subseteq \mathcal { F }$ is a finite subset.

To accommodate a limited budget, we wish to compute and store only the most important coefficients $\displaystyle c _ { i , \nu }$ contributing to the approximation of  via (4.6); we thus wish to choose the $\Lambda _ { i }$ possibly eficient to achieve maximum compression of the operator (in particular with small cardinality of $\Lambda _ { i } )$

There are two major factors at play, as described in Section 4.1.1: Both for the input x $\in \mathcal { X } ^ { s } =$ $\{ x \in H ^ { s } : x ( 0 ) = x ( 1 ) = 0 \}$ and the output $\mathcal G ( x ) \in \mathcal { Y } ^ { t } = \{ x \in H ^ { t } : x ( 0 ) = x ( 1 ) = 0 \}$ the low frequency part dominates the high frequency part. Thus input coeficients $\bar { x } _ { j }$ with lower j tend to be larger, and output coeficients $g _ { i } ( \bar { \pmb x } )$ with lower i tend to be larger. By consequence terms with large i, or with ν such that $\nu _ { j } \neq 0$ for large j, are small in (6.7). The parameters s and t encode this decay in importance: the larger s and t, the stronger the coeficient decay, the better the achievable compression. This translates to faster convergence rates for truncated polynomial expansions.

On a technical level, the Taylor expansions coeficients $\displaystyle c _ { i , \nu }$ can be bounded by investigating the domain of holomorphy of the g<sub>i</sub>, e.g., [29]. Finally, the last step of the proof of Theorem 6.13 is to reapproximate the truncated Taylor expansions (6.7) by ReLU networks; this exploits polynomial approximation results [155]. The proof of Theorem 6.13 is thus based on a careful analysis of the influence of input frequencies on frequencies of the operator output.

\- We compare the empirical convergence of spectral surrogates, PCA-Net, and FNO in notebook 5\_Comparison.ipynb. In particular, we compare these operator surrogates in terms of their cost-accuracy trade-of, where we separately consider cost contributions from training data generation, surrogate setup runtime, and surrogate evaluation runtime.

## 6.4 Algorithm unrolling

The idea of constructing deep neural networks based on iterated maps is not new. It has been used e.g. in the Telgarsky and Yarotsky constructions of ReLU approximations of polynomials [155], and for exponentially convergent expressions by ReLU NNs of certain fractal function systems with low Besov regularity in [34]. The corresponding NNs exhibit in part a recurrent structure, with the recurrent block corresponding to a ReLU emulation of the iterator map. The terminology ‘unrolling for this idea seems to have been coined in [110]. Algorithm unrolling has been leveraged for building deep solution operator surrogates of operator equations in [51]. In [158], unrolling of time-stepping schemes is combined with spatial (elliptic) solution operator surrogates to define solution operators for parabolic evolution equations. In [67], unrolled multi-grid algorithms have been used to build ONets with U-Net architecture for parametric, elliptic PDEs.

Mathematical convergence guarantees for iterative solvers of fixed-point or Newton-like iterations are generally true for open intervals of steering parameters. ONet approximations of the iteratormaps in constructive existence proofs, also in function space, or direct emulation of discretization schemes for PDEs, will imply existence of finite-parametric operator networks with convergence guarantees.

In unrolling emulations of existing, standard PDE solution algorithms with convergence guarantees and error vs. work bounds, such as for example multigrid solvers [22, 67] of Finite Element (FE) Galerkin discretizations of elliptic PDEs, the curse of dimensionality is overcome by meshindependent preconditioning and/or mesh-independent convergence rates of approximate, iterative solvers.

Existing iterative PDE solvers and constructive mathematical existence proofs are “unrolled” into deep operator networks with (sub-networks exhibiting) a recurrent structure, corresponding usually to either timestepping (for evolution equations) or to iterative (multilevel) solvers (for stationary problems).

Neural operators obtained via the unrolling approach require lower regularity of input-output maps than the holomorphic, polynomial chaos based approximation architectures. For example, for scalar hyperbolic conservation laws where e.g. the flux-to-entropy-solution dependence is known to be at best Lipschitz (see, e.g., [73]), unrolled neural operators can provide eficient flux-to-solution maps [39] which are a key component in finite-volume solvers of hyperbolic PDE.

Due to mesh-independent contraction rates of Newton- or Fixed-point-iterations, once the iterator (i.e. the operator which realizes one Newton- or Fixed-point-iteration step) has been learned, to suficient accuracy, the depth of unrolled neural operators scales polylogarithmically with the target accuracy.

Examples for this approach are, for elliptic PDEs, the “MGNet” architecture [22], [103, 104] obtained by unrolling multi-grid solution algorithms, and neural operators obtained from unrolling of time-stepping schemes for nonlinear evolution equations [39, 26]. A framework for leveraging constructive mathematical existence proofs (e.g. via fixed-point or Newton-like iterations in function space) resulting in deep recurrent surrogate architectures is developed in [51]. Examples of neural operators built by unrolling PDE solution algorithms are available for hyperbolic PDEs [39], parabolic PDEs [38, 98, 26], and elliptic PDEs [103, 104, 22].

## 7 Generalization Errors

While an analysis of approximation errors yields theoretically achievable convergence rates for certain types of architectures, these rates are not necessarily observed when training an architecture to minimize an empirical risk, e.g., [62, 61]. Exceptions include interpolation based architectures (which do not necessarily require risk minimization, see Section 4.1.2) and also architectures with linear parameter dependence, e.g. [11, 2].

Let $\mu$ be a measure on $\mathcal { X }$ and suppose $x _ { 1 } , \ldots , x _ { m } \stackrel { \mathrm { i i d } } { \sim } \mu$ are random samples. Given data pairs $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { m }$ generated via (5.1), as explained in Section 5, during training the empirical risk (5.2) is minimized. Denote such an empirical risk minimizer by $\hat { \mathcal { G } } _ { \theta }$ . Our ultimate goal is to achieve a small population risk, which is the expected squared error over the true data distribution $\mu$

$$
\lVert \mathcal { G } ( x ) - \hat { \mathcal { G } } _ { \boldsymbol { \theta } } ( x ) \rVert _ { L ^ { 2 } ( \mu ) } ^ { 2 } ,\tag{7.1}
$$

also referred to as the population risk. The question of generalization errors is the following: Is the empirical risk close to the population risk? Intuitively we assume this to be true, if the amount of data samples m in (5.2) is large enough, since (5.2) can be interpreted as an approximation to (7.1). This implies in particular that a small empirical risk yields a small population risk. In this case the approximation $\hat { \mathcal { G } } _ { \theta }$ of the operator $\mathcal { G }$ generalizes well to unseen data outside of our training sample $x _ { 1 } , \ldots , x _ { m } .$

We next recall a statement of this type from [128, Theorem 3.15], which is in the same setting as the approximation result Theorem 6.13; in particular $\mathcal { G }$ is a holomorphic operator.

Theorem 7.1. Let $x , y$ be two separable Hilbert spaces, let ,  be as in Section 6.3.1 and let $\delta > 0$ . Suppose further $\mathcal { G } : \mathcal { X }  \mathcal { Y }$ satisfies Assumption 6.11 for some $s > 1 , t > 0$ , and assume that $\mu$ is a probability measure on such that

$$
\operatorname { s u p p } ( { \boldsymbol { \mu } } ) \leq \left\{ x \in { \mathcal { X } } : \operatorname* { s u p } _ { j \in \mathbb { N } } j ^ { t } | \varphi _ { { \mathcal { X } } , j } ( x ) | \leq R \right\} .
$$

Let data $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { m }$ be generated by (5.1), where $\varepsilon _ { i }$ are i.i.d. draws from some subgaussian<sup>8</sup> noise in . There exists a ReLU neural network architecture, such that a network ${ \hat { g } } _ { \theta }$ minimizing the empirical risk

$$
\frac { 1 } { m } \sum _ { j = 1 } ^ { m } \| \mathcal { D } \circ \widehat { g } _ { \boldsymbol { \theta } } \circ \mathcal { E } ( x _ { j } ) - y _ { j } \| _ { \mathcal { V } } ^ { 2 } ,
$$

satisfies with $\kappa = \operatorname* { m i n } \{ s - 1 , t \}$

$$
\begin{array} { r } { \mathbb { E } [ \| \mathcal { D } \circ \widehat { g } _ { \theta } \circ \mathcal { E } - \mathcal { G } \| _ { L ^ { 2 } ( \mu ) } ^ { 2 } ] \leq C m ^ { - \frac { 2 \kappa } { 2 \kappa + 2 } + \delta } . } \end{array}
$$

Here the constant C depends on δ, but is independent of $m ,$ and the expectation is over the random choice of $x _ { i } ,$ and noise terms $\varepsilon _ { i } , f o r i = 1 , \dots , m$

We point out that κ in Theorem 7.1 corresponds to the approximation rate in Theorem 6.13; for holomorphic operators, asymptotically, it is solely governed by the algebraic decay of the input (s) and output (t) coeficients. As discussed, these parameters describe the intrinsic dimensionality of the problem, or more precisely, they determine how fast the latent input/output dimensions need to increase as the desired accuracy increases. As s, t jointly, the convergence rate $\frac { 2 \kappa } { 2 \kappa + 2 }$ in Theorem 7.1 gets arbitrarily close to 1.

Remark 7.2. As mentioned in Remark 5.1, it is common to model the noise not as subgaussian, but as white noise. The statement also extends to this case, but with a more technical formulation.

The theorem gives convergence rates for empirical risk minimizers trained on noisy data. In particular it shows that holomorphic operators can be learned with moderate amounts of data. Its proof is based on adapting standard techniques from M-estimation and empirical process theory to the case of operators [42, 147]. While we give a simplified version postulating only existence of a suitable network, the precise architecture of $g _ { \theta }$ is given in [128]. Importantly, it corresponds to a sparsely connected deep neural network (hence, with nonlinear parameter dependence); the terminology “sparsely connected” refers to a neural network architecture for which various weights are apriorily set to zero, and which are not adapted during training (i.e., do not belong to the set of trainable parameters). The theorem remains true also for fully connected ReLU networks, albeit with a worse convergence rate. For neural networks for which only the weights in the last layer are trained (hence, with linear parameter dependence), related results can also be found in [2, 3].

Moreover, we have already seen that weaker regularity assumptions in general lead to a curse of dimensionality in terms of the required data, see Proposition 6.9. For Lipschitz operators this is discussed in more detail in [96], where statistical convergence rates are derived. For lower bounds we refer once more to [80], and for the FNO architecture, some results regarding empirical risk minimization without noisy data are discussed in [78]. Finally, we also refer to [4] for recent results on minimax rates for the class of Hölder continuous operators.

## 8 Conclusion and Further Directions

Parsimonious operator surrogates have seen and continue to see at the time of this writing a very dynamic development, both on the algorithmic, as well as on the theoretical side. They emerge as one key building block of SciML, in rather diverse computational tasks as, e.g., optimal design, shape-optimization [102, 64], Bayesian Inversion [18], and the currently active area of Digital Twins [97, 142, 75, 141].

Universality of neural operators has been established for a wide range of architectures in [95, 133, 86, 1, 57].

Currently, approximation rate theory in smoothness scales is available only in few cases. A broad class of operators which aford expression rates free from the CoD are holomorphic maps [29, 30, 27, 87, 7, 74, 71, 128]. Here, neural network based surrogates are, essentially, equivalent to other classes of approximators $g ^ { N }$ in (4.1), such as spectral surrogates as in [50, 71].

Expression rate results are also available for neural operator surrogates built from ‘unrolling existing PDE solvers into neural network architectures. We refer to [41, 40] for a survey of results in this direction, to [51], and to the discussion in Section 6.4.

An important area not covered here is the training error, when neural operator architectures are calibrated with i.i.d. training data. For general theory of neural network training we refer, e.g., to the recent [105]. Of particular interest is enhancing eficiency in Operator Network training in architectures (3.1) with multi-level wavelet- or frame-encoders and -decoders, as outlined in Example 3.3 item four, and Figure 1. Multi-level encoders and decoders naturally induce a multi-level structure in the parametric approximator map $g ^ { N }$ in (4.1), with diferent components of the parametric surrogates $g ^ { N }$ requiring diferent accuracy in order to ensure a prescribed accuracy of the associated neural operator $\hat { \mathcal { G } } ^ { N }$ in (4.1), as map between function spaces. This is naturally accommodated in the presented framework by suitable choices of weight-sequences in (6.4), see also [71] for more general choices of weight sequences than those in (6.4). In the context of Neural Operator Training, multi-level splitting of input- and output-spaces is leveraged for eficiency in [131].

Approximation architectures other than (generalized) polynomial chaos have been explored for operator learning, in particular for dynamical systems in high- or infinite-dimensional state- and parameter spaces. We mention only kernel methods e.g. [11, 153], and Gaussian processes e.g. [23] and the references there, and random-feature type neural approximations in solution representation formulas, e.g. for initial-value problems for wave-equations [25]. Detailed discussion and comparisons are beyond the scope of this chapter.

Finally, we emphasize that the performance of diferent approaches and models can difer significantly depending on the problem; careful numerical experiments and comparisons are required to identify suitable architectures that are computationally eficient for specific applications. We refer to [100, 37, 47] for several recent works in this direction. Moreover, [151] provides an empirical comparison between neural and polynomial operator surrogates. This study observes that spectral methods typically outperform learning-based approaches in regimes of high regularity and low intrinsic parametric dimensionality. In other regimes, neural networks and learning-based approaches may be capable of detecting structures and forms of compression that appear unreachable by direct polynomial methods. Ultimately, however, these comparative advantages remain heavily problem-dependent.

## References

[1] B. Acciaio, A. Kratsios, and G. Pammer. Designing universal causal deep learning models: the geometric (hyper)transformer. Math. Finance, 34(2):671–735, 2024.

[2] B. Adcock, S. Brugiapaglia, N. Dexter, and S. Moraga. Near-optimal learning of banach-valued, highdimensional functions via deep neural networks. Neural Networks, 181:106761, 2025.

[3] B. Adcock, N. Dexter, and S. Moraga. Optimal deep learning of holomorphic operators between Banach spaces. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 27725–27789. Curran Associates, Inc., 2024.

[4] B. Adcock, G. Maier, and R. Parhi. Towards sharp minimax risk bounds for operator learning, 2025.

[5] M. Asch. A toolbox for digital twins—from model-based to data-driven, volume 6 of Mathematics in Industry (Philadelphia). Society for Industrial and Applied Mathematics (SIAM), Philadelphia, PA, [2022] ©2022.

[6] M. Asch, M. Bocquet, and M. Nodet. Data assimilation, volume 11 of Fundamentals of Algorithms. Society for Industrial and Applied Mathematics (SIAM), Philadelphia, PA, 2016. Methods, algorithms, and applications.

[7] R. Aylwin, C. Jerez-Hanckes, C. Schwab, and J. Zech. Domain uncertainty quantification in computational electromagnetics. SIAM/ASA J. Uncertainty Quantification, 8(1):301–341, 2020.

[8] M. Bachmayr and V. K. Nguyen. Identifiability of difusion coeficients for source terms of non-uniform sign. Inverse Probl. Imaging, 13(5):1007–1021, 2019.

[9] G. Bai, U. Koley, S. Mishra, and R. Molinaro. Physics informed neural networks (PINNs) for approximating nonlinear dispersive PDEs. J. Comput. Math., 39(6):816–847, 2021.

[10] R. Baptista, A. Stuart, and S. Tran. Large language models: A mathematical formulation, 2026.

[11] P. Batlle, M. Darcy, B. Hosseini, and H. Owhadi. Kernel methods are competitive for operator learning. Journal of Computational Physics, 496:112549, 2024.

[12] F. E. Benth, N. Detering, and L. Galimberti. Neural networks in fréchet spaces, 2022.

[13] K. Bhattacharya, B. Hosseini, N. B. Kovachki, and A. M. Stuart. Model Reduction And Neural Networks For Parametric PDEs. The SMAI Journal of computational mathematics, 7:121–157, 2021.

[14] V. I. Bogachev. Measure theory. Vol. I, II. Springer-Verlag, Berlin, 2007.

[15] S. Brivio, S. Fresca, N. R. Franco, and A. Manzoni. Error estimates for pod-dl-roms: a deep learning framework for reduced order modeling of nonlinear parametrized pdes enhanced by proper orthogonal decomposition, 2023.

[16] H.-J. Bungartz and M. Griebel. Sparse grids. Acta Numerica, 13:147–269, 2004.

[17] E. Calvello, N. B. Kovachki, M. E. Levine, and A. M. Stuart. Continuum attention for neural operators, 2025.

[18] L. Cao, T. O’Leary-Roseberry, and O. Ghattas. Derivative-informed neural operator acceleration of geometric MCMC for infinite-dimensional Bayesian inverse problems. J. Mach. Learn. Res., 26:Paper No. [78], 68, 2025.

[19] S. Cao. Choose a transformer: Fourier or galerkin, 2021.

[20] D. Chae and C. Foiaş. On the homogeneous statistical solutions of the 2-D Navier-Stokes equations. Indiana Univ. Math. J., 43(1):177–185, 1994.

[21] T. Chen and H. Chen. Approximations of continuous functionals by neural networks with application to dynamic systems. IEEE Transactions on Neural Networks, 4:910 – 918, 1993.

[22] Y. Chen, B. Dong, and J. Xu. Meta-MgNet: meta multigrid networks for solving parameterized partial diferential equations. J. Comput. Phys., 455:Paper No. 110996, 26, 2022.

[23] Y. Chen, B. Hosseini, H. Owhadi, and A. M. Stuart. Solving and learning nonlinear PDEs with Gaussian processes. J. Comput. Phys., 447:Paper No. 110668, 29, 2021.

[24] W. Cho, K. Lee, N. Park, D. Rim, and G. Welper. FastLRNR and sparse physics informed backpropagation. Results Appl. Math., 25:Paper No. 100547, 7, 2025.

[25] W. Cho, K. Lee, N. Park, D. Rim, and G. Welper. Learning low rank neural representations of hyperbolic wave dynamics from data, 2025.

[26] J. Choi, T.-Y. Chang, N. Kim, and Y. Hong. A data free neural operator enabling fast inference of 2D and 3D Navier Stokes equations, 2025.

[27] A. Cohen and R. DeVore. Approximation of high-dimensional parametric pdes. Acta Numerica, 24:1– 159, 2015.

[28] A. Cohen and R. DeVore. Kolmogorov widths under holomorphic mappings. IMA J. Numer. Anal., 36(1):1–12, 2016.

[29] A. Cohen, R. Devore, and C. Schwab. Analytic regularity and polynomial approximation of parametric and stochastic elliptic PDE’s. Anal. Appl. (Singap.), 9(1):11–47, 2011.

[30] A. Cohen, C. Schwab, and J. Zech. Shape holomorphy of the stationary Navier-Stokes Equations. SIAM J. Math. Analysis, 50(2):1720–1752, 2018.

[31] P. G. Constantine. Active subspaces, volume 2 of SIAM Spotlights. Society for Industrial and Applied Mathematics (SIAM), Philadelphia, PA, 2015. Emerging ideas for dimension reduction in parameter studies.

[32] W. M. Czarnecki, S. Osindero, M. Jaderberg, G. Swirszcz, and R. Pascanu. Sobolev training for neural networks. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, pages 4281–4290, Red Hook, NY, USA, 2017. Curran Associates Inc.

[33] W. Dahmen, A. Kunoth, and K. Urban. Biorthogonal spline-wavelets on the interval – stability and moment conditions. Appl. Comp. Harm. Anal., 6:259–302, 1999.

[34] I. Daubechies, R. DeVore, S. Foucart, B. Hanin, and G. Petrova. Nonlinear approximation and (deep) ReLU networks. Constr. Approx., 55(1):127–172, 2022.

[35] P. J. Davis. Interpolation and approximation. Dover Publications, Inc., New York, 1975. Republication, with minor corrections, of the 1963 original, with a new preface and bibliography.

[36] O. Davydov and R. Stevenson. Hierarchical Riesz bases for $H ^ { s } ( \Omega ) , \ 1 < \ s < \ \frac { 5 } { 2 } .$ . Constr. Approx., 22(3):365–394, 2005.

[37] M. De Hoop, D. Zhengyu, Huang, E. Qian, and A. Stuart, M. The cost-accuracy trade-of in operator learning with neural networks. Journal of Machine Learning, 1(3):299–341, 2022.

[38] T. De Ryck, A. D. Jagtap, and S. Mishra. Error estimates for physics-informed neural networks approximating the Navier-Stokes equations. IMA J. Numer. Anal., 44(1):83–119, 2024.

[39] T. De Ryck and S. Mishra. Error analysis for deep neural network approximations of parametric hyperbolic conservation laws. Math. Comp., 93(350):2643–2677, 2024.

[40] T. De Ryck and S. Mishra. Numerical analysis of physics-informed neural networks and related models in physics-informed machine learning. Acta Numer., 33:633–713, 2024.

[41] B. Deng, Y. Shin, L. Lu, Z. Zhang, and G. E. Karniadakis. Convergence rate of Deep ONets for learning operators arising from advection-difusion equations. Neural Networks, 153:411–426, 2022.

[42] S. Dirksen. Tail bounds via generic chaining. Electronic Journal of Probability, 20, 2015.

[43] S. Dolgov and R. Scheichl. A hybrid alternating least squares–tt-cross algorithm for parametric pdes. SIAM/ASA Journal on Uncertainty Quantification, 7(1):260–291, 2019.

[44] S. Dong and Z. Li. Local extreme learning machines and domain decomposition for solving linear and nonlinear partial diferential equations. Comput. Methods Appl. Mech. Engrg., 387:Paper No. 114129, 63, 2021.

[45] D. Dung, V. K. Nguyen, C. Schwab, and J. Zech. Analyticity and sparsity in uncertainty quantification for pdes with gaussian random field inputs. Springer Lecture Notes in Mathematics, 2334, 2023.

[46] A. Ern and J.-L. Guermond. Theory and Practice of Finite Elements, volume 159 of Applied Mathematical Sciences. Springer, New York, 2004.

[47] O. G. Ernst, B. Sprungk, and C. Zhang. Uncertainty modeling and propagation for groundwater flow: a comparative study of surrogates. GEM Int. J. Geomath., 15(1):Paper No. 11, 41, 2024.

[48] Ernst, Oliver G., Mugler, Antje, Starklof, Hans-Jörg, and Ullmann, Elisabeth. On the convergence of generalized polynomial chaos expansions. ESAIM: M2AN, 46(2):317–339, 2012.

[49] L. C. Evans. Partial diferential equations. American Mathematical Society, Providence, R.I., 2010.

[50] V. S. Fanaskov and I. V. Oseledets. Spectral neural operators. Dokl. Math., 108(suppl. 2):S226–S232, 2023. Translation of Dokl. Akad. Nauk 514 (2023), no. 2, 72–79.

[51] M. Feischl, C. Schwab, and F. Zehetgruber. Neural General Operator Networks via Banach Fixed Point Iterations. Technical Report 2025-13, Seminar for Applied Mathematics, ETH Zürich, Switzerland, 2025.

[52] C. Foiaş, O. Manley, R. Rosa, and R. Temam. Navier-Stokes equations and turbulence, volume 83 of Encyclopedia of Mathematics and its Applications. Cambridge University Press, Cambridge, 2001.

[53] N. R. Franco, S. Fresca, A. Manzoni, and P. Zunino. Approximation bounds for convolutional neural networks in operator learning. Neural Networks, 161:129–141, 2023.

[54] N. R. Franco, A. Manzoni, and P. Zunino. A deep learning approach to reduced order modelling of parameter dependent partial diferential equations. Mathematics of Computation, 92, 11 2022.

[55] T. Furuya, M. V. de Hoop, and G. Peyré. Transformers are universal in-context learners, 2024.

[56] S. Ghosh, A. Opala, M. Matuszewski, T. Paterek, and T. C. H. Liew. Quantum reservoir processing. npj Quantum Information, 5(1), Apr. 2019.

[57] J. Gödeke and P. Fernsel. New universal operator approximation theorem for encoder-decoder architectures (preprint), 2025.

[58] L. Gonon. Random feature neural networks learn Black-Scholes type PDEs without curse of dimensionality. J. Mach. Learn. Res., 24:Paper No. [189], 51, 2023.

[59] L. Gonon and A. Jacquier. Universal approximation theorem and error bounds for quantum neural networks and quantum reservoirs. IEEE Trans. Neural Netw. Learn. Syst., 36(6):11355–11368, 2025.

[60] K. Gröchenig. Localization of frames, Banach frames, and the invertibility of the frame operator. J. Fourier Anal. Appl., 10(2):105–132, 2004.

[61] P. Grohs, S. Lanthaler, and M. Trautner. Theory-to-practice gap for neural networks and neural operators, 2025.

[62] P. Grohs and F. Voigtlaender. Proof of the theory-to-practice gap in deep learning via sampling complexity bounds for neural network approximation spaces. Foundations of Computational Mathematics, 24(4):1085–1143, Aug 2024.

[63] H. Harbrecht, R. Schneider, and C. Schwab. Multilevel frames for sparse tensor product spaces. Numer. Math., 110(2):199–220, 2008.

[64] H. Harbrecht and C. Schwab. Neural shape operator surrogates – expression rate bounds, 2026.

[65] H. Harbrecht, C. Schwab, and M. Zank. Wavelet compressed, modified Hilbert transform in the space-time discretization of the heat equation, 2025.

[66] C. Heil. A basis theory primer. Applied and Numerical Harmonic Analysis. Birkhäuser/Springer, New York, expanded edition, 2011.

[67] C. Heiß, I. Gühring, and M. Eigel. Multilevel CNNs for parametric PDEs. J. Mach. Learn. Res., 24:Paper No. [373], 42, 2023.

[68] F. Henríquez and C. Schwab. Shape Holomorphy of the Calderón Projector for the Laplacian in R<sup>2</sup>. Integral Equations and Operator Theory, 93(4):43, Jul 2021.

[69] M. Herde, B. Raonić, T. Rohner, R. Käppeli, R. Molinaro, E. de Bézenac, and S. Mishra. Poseidon: eficient foundation models for pdes. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA, 2025. Curran Associates Inc.

[70] L. Herrmann, C. Schwab, and J. Zech. Uncertainty quantification for spectral fractional difusion: sparsity analysis of parametric solutions. SIAM/ASA J. Uncertain. Quantif., 7(3):913–947, 2019.

[71] L. Herrmann, C. Schwab, and J. Zech. Neural and spectral operator surrogates: unified construction and expression rate bounds. Adv. Comput. Math., 50(4):Paper No. 72, 43, 2024.

[72] J. Hesthaven and S. Ubbiali. Non-intrusive reduced order modeling of nonlinear problems using neural networks. Journal of Computational Physics, 363:55–78, 2018.

[73] H. Holden and N. H. Risebro. Front tracking for hyperbolic conservation laws, volume 152 of Applied Mathematical Sciences. Springer, New York, 2011. First softcover corrected printing of the 2002 original.

[74] C. Jerez-Hanckes, C. Schwab, and J. Zech. Electromagnetic wave scattering by random surfaces: shape holomorphy. Math. Models Methods Appl. Sci., 27(12):2229–2259, 2017.

[75] M. Kapteyn, A. Chaudhuri, E. A. B. F. Lima, G. Pash, R. Bravo, K. Willcox, T. E. Yankeelov, and D. A. H. II. Tumortwin: A python framework for patient-specific digital twins in oncology, 2025.

[76] W. A. Klimke. Uncertainty modeling using fuzzy arithmetic and sparse grids. Phd thesis, Universität Stuttgart, 2006. Available at CiteSeerX: https://citeseerx.ist.psu.edu/document?repid=rep1& type=pdf&doi=9da00c0aedf5335927147df78eb5f6767edba968.

[77] A. N. Kolmogorov. On the representations of continuous functions of many variables by superposition of continuous functions of one variable and addition. In Dokl. Akad. Nauk USSR, volume 114, pages 953–956, 1957.

[78] N. Kovachki, S. Lanthaler, and S. Mishra. On universal approximation and error bounds for Fourier neural operators. Journal of Machine Learning Research, 22(290):1–76, 2021.

[79] N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. Stuart, and A. Anandkumar. Neural operator: learning maps between function spaces with applications to PDEs. J. Mach. Learn. Res., 24:Paper No. [89], 97, 2023.

[80] N. B. Kovachki, S. Lanthaler, and H. Mhaskar. Data complexity estimates for operator learning, 2024.

[81] N. B. Kovachki, S. Lanthaler, and A. M. Stuart. Operator learning: Algorithms and analysis, 2024.

[82] A. Kratsios, C. Liu, M. Lassas, M. V. de Hoop, and I. Dokmanic. An approximation theory for metric space-valued functions with a view towards deep learning, 2023.

[83] F. Laakmann and P. Petersen. Eficient approximation of solutions of parametric linear transport equations by ReLU DNNs. Adv. Comput. Math., 47(1):Paper No. 11, 32, 2021.

[84] S. Lanthaler. Operator learning with PCA-Net: upper and lower complexity bounds. J. Mach. Learn. Res., 24:Paper No. [318], 67, 2023.

[85] S. Lanthaler. Operator learning of lipschitz operators: An information-theoretic perspective, 2024.

[86] S. Lanthaler, Z. Li, and A. M. Stuart. Nonlocality and Nonlinearity Implies Universality in Operator Learning. arXiv e-prints, page arXiv:2304.13221, Apr. 2023.

[87] S. Lanthaler, S. Mishra, and G. E. Karniadakis. Error estimates for DeepONets: a deep learning framework in infinite dimensions. Trans. Math. Appl., 6(1):tnac001, 141, 2022.

[88] S. Lanthaler and N. H. Nelsen. Error bounds for learning with vector-valued random features, 2023.

[89] S. Lanthaler, A. M. Stuart, and M. Trautner. Discretization error of fourier neural operators, 2024.

[90] V. Lebedev, Y. Ganin, M. Rakhuba, I. Oseledets, and V. Lempitsky. Speeding-up convolutional neural networks using fine-tuned cp-decomposition, 2015.

[91] J. Lebl. Tasty bits of several complex variables. Independently published, Version 4.2, May 20th, 2025.

[92] F. Leonardi, S. Mishra, and C. Schwab. Numerical approximation of statistical solutions of planar, incompressible flows. Math. Models Methods Appl. Sci., 26(13):2471–2523, 2016.

[93] M. Leshno, V. Y. Lin, A. Pinkus, and S. Schocken. Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. Neural Networks, 6(6):861–867, 1993.

[94] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar. Multipole graph neural operator for parametric partial diferential equations, 2020.

[95] Z. Li, N. B. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar. Fourier neural operator for parametric partial diferential equations. International Conference on Learning Representations, 2021.

[96] H. Liu, H. Yang, M. Chen, T. Zhao, and W. Liao. Deep nonparametric estimation of operators between infinite dimensional spaces. Journal of Machine Learning Research, 25(24):1–67, 2024.

[97] N. Liu, X. Li, M. R. Rajanna, E. W. Reutzel, B. Sawyer, P. Rao, J. Lua, N. Phan, and Y. Yu. Deep neural operator enabled digital twin modeling for additive manufacturing, 2024.

[98] B. Lorenz, A. Bacho, and G. Kutyniok. Error estimation for physics-informed neural networks approximating semilinear wave equations, 2024.

[99] M. Lowery, J. Turnage, Z. Morrow, J. D. Jakeman, A. Narayan, S. Zhe, and V. Shankar. Kernel neural operators (knos) for scalable, memory-eficient, geometrically-flexible operator learning, 2026.

[100] L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, and G. E. Karniadakis. A comprehensive and fair comparison of two neural operators (with practical extensions) based on FAIR data. Comput. Methods Appl. Mech. Engrg., 393:Paper No. 114778, 35, 2022.

[101] D. Luo, T. O’Leary-Roseberry, P. Chen, and O. Ghattas. Dimension reduction for derivative-informed operator learning: An analysis of approximation errors, 2025.

[102] D. Luo, T. O’Leary-Roseberry, P. Chen, and O. Ghattas. Eficient PDE-constrained optimization under high-dimensional uncertainty using derivative-informed neural operators. SIAM J. Sci. Comput., 47(4):C899–C931, 2025.

[103] C. Marcati and C. Schwab. Exponential convergence of deep operator networks for elliptic partial diferential equations. SIAM J. Numer. Anal., 61(3):1513–1545, 2023.

[104] C. Marcati and C. Schwab. Expression Rates of Neural Operators for Linear Elliptic PDEs in Polytopes. Technical Report 2024-23, Seminar for Applied Mathematics, ETH Zürich, Switzerland, 2024.

[105] S. Marcotte, G. Peyré, and R. Gribonval. Intrinsic training dynamics of deep neural networks, 2025.

[106] H. N. Mhaskar. Local approximation of operators. Appl. Comput. Harmon. Anal., 64:194–228, 2023.

[107] H. N. Mhaskar and N. Hahm. Neural networks for functional approximation and system identification. Neural Comput, 9(1):143–159, Jan. 1997.

[108] C. Milbradt and M. Wahl. High-probability bounds for the reconstruction error of pca. Statistics & Probability Letters, 161:108741, 2020.

[109] R. Molinaro, S. Lanthaler, B. Raonic, T. Rohner, V. Armegioiu, S. Simonis, D. Grund, Y. Ramic, Z. Y. Wan, F. Sha, S. Mishra, and L. Zepeda-Núnez. Generative ai for fast and accurate statistica computation of fluids, 2025.

[110] V. Monga, Y. Li, and Y. C. Eldar. Algorithm unrolling: Interpretable, eficient deep learning for signal and image processing. IEEE Signal Processing Magazine, 38:18–44, 2021.

[111] G. A. Muñoz, Y. Sarantopoulos, and A. Tonge. Complexifications of real Banach spaces, polynomials and multilinear maps. Studia Math., 134(1):1–33, 1999.

[112] N. H. Nelsen and A. M. Stuart. Operator learning using random features: a tool for scientific computing. SIAM Rev., 66(3):535–571, 2024. Revised reprint of “The random feature model for input-output maps between Banach spaces” [ 4313849].

[113] H. Nguyen and R. Stevenson. Finite element wavelets with improved quantitative properties. J. Comput. Appl. Math., 230(2):706–727, 2009.

[114] M. Nguyen and N. Mücke. Random features for operator-valued kernels: Bridging kernel methods and neural operators, 2026.

[115] T. O’Leary-Roseberry, P. Chen, U. Villa, and O. Ghattas. Derivative-informed neural operator: An eficient framework for high-dimensional parametric derivative learning. Journal of Computational Physics, 496:112555, 2024.

[116] T. O’Leary-Roseberry, U. Villa, P. Chen, and O. Ghattas. Derivative-informed projected neural networks for high-dimensional parametric maps governed by pdes. Computer Methods in Applied Mechanics and Engineering, 388:114199, 2022.

[117] J. A. A. Opschoor and C. Schwab. Deep ReLU networks and high-order finite element methods II: Chebyšev emulation. Comput. Math. Appl., 169:142–162, 2024.

[118] J. A. A. Opschoor, C. Schwab, and J. Zech. Exponential ReLU DNN expression of holomorphic maps in high dimension. Constr. Approx., 55(1):537–582, 2022.

[119] I. Oseledets and E. Tyrtyshnikov. Tt-cross approximation for multidimensional arrays. Linear Algebra and its Applications, 432:70–88, 01 2010.

[120] I. V. Oseledets. Tensor-train decomposition. SIAM Journal on Scientific Computing, 33(5):2295–2317, 2011.

[121] W. Ou, P. Schenkel, and H. Boelcskei. Three Quantization Regimes for ReLU Networks, 2024.

[122] G. Pash, U. Villa, D. A. H. II, T. E. Yankeelov, and K. Willcox. Predictive digital twins with quantified uncertainty for patient-specific decision making in oncology, 2025.

[123] P. Petersen and J. Zech. Mathematical theory of deep learning, 2025.

[124] R. H. R. DeVore and C. Micchelli. Optimal nonlinear approximation. Manuscripta mathematica, 63(4):469–478, 1989.

[125] B. Raonic, R. Molinaro, T. De Ryck, T. Rohner, F. Bartolucci, R. Alaifari, S. Mishra, and E. de Bézenac. Convolutional neural operators for robust and accurate learning of pdes. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 77187–77200. Curran Associates, Inc., 2023.

[126] N. Rares Franco, A. Manzoni, and P. Zunino. A deep learning approach to reduced order modelling of parameter dependent partial diferential equations. Mathematics of Computation, 92(340):483–524, Nov. 2022.

[127] F. Regazzoni, S. Pagani, and A. Quarteroni. Universal solution manifold networks (usm-nets): nonintrusive mesh-free surrogate models for problems in variable domains, 2022.

[128] N. Reinhardt, S. Wang, and J. Zech. Statistical learning theory for neural operators, 2024.

[129] M. Reiß and M. Wahl. Nonasymptotic upper bounds for the reconstruction error of PCA. The Annals of Statistics, 48(2):1098–1123, apr 2020.

[130] N. Rekatsinas and R. Stevenson. A quadratic finite element wavelet Riesz basis. Int. J. Wavelets Multiresolut. Inf. Process., 16(4):1850033, 17, 2018.

[131] J. Rowbottom, S. Fresca, P. Lio, C.-B. Schönlieb, and N. Boullé. Multi-Level Monte Carlo training of neural operators. Comput. Methods Appl. Mech. Engrg., 453:Paper No. 118800, 17, 2026.

[132] D. Schötzau and C. Schwab. An hp a priori error analysis of the DG time-stepping method for initial value problems. Calcolo, 37(4):207–232, 2000.

[133] C. Schwab, A. Stein, and J. Zech. Deep operator network approximation rates for Lipschitz operators. Analysis and Applications, 24(01):199–239, 2026.

[134] C. Schwab and R. Stevenson. Space-time adaptive wavelet methods for parabolic evolution problems. Math. Comp., 78(267):1293–1318, 2009.

[135] C. Schwab and R. Stevenson. Fractional space-time variational formulations of (Navier-) Stokes equations. SIAM J. Math. Anal., 49(4):2442–2467, 2017.

[136] B. Shih, A. Peyvan, Z. Zhang, and G. E. Karniadakis. Transformers as Neural Operators for Solutions of Diferential Equations with Finite Regularity, 2024.

[137] R. C. Smith. Uncertainty quantification—theory, implementation, and applications, volume 30 of Computational Science & Engineering. Society for Industrial and Applied Mathematics (SIAM), Philadelphia, PA, [2024] ©2024. Second edition [of 3155184].

[138] E. M. STEIN. Singular Integrals and Diferentiability Properties of Functions (PMS-30). Princeton University Press, 1970.

[139] I. Steinwart and C. Scovel. Mercer’s theorem on general domains: on the interaction between measures, kernels, and RKHSs. Constr. Approx., 35(3):363–417, 2012.

[140] M. Telgarsky. Benefits of depth in neural networks. In V. Feldman, A. Rakhlin, and O. Shamir, editors, 29th Annual Conference on Learning Theory, volume 49 of Proceedings of Machine Learning Research, pages 1517–1539, Columbia University, New York, New York, USA, 23–26 Jun 2016. PMLR.

[141] M. Tezzele, S. Carr, U. Topcu, and K. E. Willcox. Adaptive planning for risk-aware predictive digital twins, 2024.

[142] M. Torzoni, M. Tezzele, S. Mariani, A. Manzoni, and K. E. Willcox. A digital twin framework for civil engineering structures. Comput. Methods Appl. Mech. Engrg., 418(part B):Paper No. 116584, 20, 2024.

[143] H. Triebel. Function Spaces and Wavelets on Domains, volume 7 of EMS Tracts in Mathematics. European Mathematical Society (EMS), Zürich, 2008.

[144] H. Triebel. Bases in function spaces, sampling, discrepancy, numerical integration, volume 11 of EMS Tracts in Mathematics. European Mathematical Society (EMS), Zürich, 2010.

[145] R. Tu, C. White, J. Kossaifi, B. Bonev, N. Kovachki, G. Pekhimenko, K. Azizzadenesheli, and A. Anandkumar. Guaranteed approximation bounds for mixed-precision neural operators, 2024.

[146] K. Urban. Wavelet Methods for Elliptic Partial Diferential Equations. Numerical Mathematics and Scientific Computation. Oxford University Press, Oxford, 2009.

[147] S. van de Geer. Empirical Processes in M-Estimation. Cambridge U. Press, 2000.

[148] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need, 2017.

[149] W. Walter. Ordinary diferential equations, volume 182 of Graduate Texts in Mathematics. Springer-Verlag, New York, 1998. Translated from the sixth German (1996) edition by Russell Thompson, Readings in Mathematics.

[150] S. Wen, A. Kumbhat, L. Lingsch, S. Mousavi, Y. Zhao, P. Chandrashekar, and S. Mishra. Geometry aware operator transformer as an eficient and accurate neural surrogate for pdes on arbitrary domains, 2025.

[151] J. Westermann, B. Huber, T. O’Leary-Roseberry, and J. Zech. Performance of neural and polynomial operator surrogates, 2026.

[152] D. Xiu and G. E. Karniadakis. The wiener–askey polynomial chaos for stochastic diferential equations. SIAM Journal on Scientific Computing, 24(2):619–644, 2002.

[153] L. Yang, X. Sun, B. Hamzi, H. Owhadi, and N. Xie. Learning dynamical systems from data: a simple cross-validation perspective, part V: Sparse kernel flows for 132 chaotic dynamical systems. Phys. D, 460:Paper No. 134070, 13, 2024.

[154] B. Yao, D. Luo, L. Cao, N. Kovachki, T. O’Leary-Roseberry, and O. Ghattas. Derivative-Informed Fourier Neural Operator: Universal Approximation and Applications to PDE-Constrained Optimization, 2025.

[155] D. Yarotsky. Error bounds for approximations with deep ReLU networks. Neural Netw., 94:103–114, 2017.

[156] J. Zech. Sparse-Grid Approximation of High-Dimensional Parametric PDEs. PhD thesis, ETH Zürich, 2018.

[157] E. Zeidler. Nonlinear Functional Analysis and Its Applications. Number Bd. 1 in Nonlinear functional analysis and its applications. Springer, 1986.

[158] J. Zhao, N. Panda, Y. T. Lin, E. Valseth, D. Oyen, and C. Dawson. HyCOP: Hybrid Composition Operators for Interpretable Learning of PDEs, 2026.