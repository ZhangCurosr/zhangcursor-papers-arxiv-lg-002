# SOLITON PROFILES IN 1D: ENHANCED NEURAL-NETWORK SOLVERS AND CONSTRAINED DEEP RITZ METHODS

CHANDLER HAIGHT, ALEX DAVID RODRIGUEZ, AND SVETLANA ROUDENKO

Abstract. We develop and compare enhanced neural-network solvers for one-dimensional ground-state profiles, which are solutions of

$$
Q ^ { \prime \prime } = b Q - \gamma Q ^ { p } , \quad Q > 0 , \quad p > 1 , \quad b , \gamma > 0 ,
$$

the stationary profile equation for the nonlinear Schr¨odinger, generalized Korteweg–de Vries, and nonlinear Klein–Gordon equations. Its explicit sech-type solutions serve as exact reference for error computation and benchmark hyperparameter selection. We improve on the standard Adam-trained physics-informed neural network (PINN) baseline by introducing higher-order optimization via Gauss–Newton/Levenberg– Marquardt and L–BFGS refinements, as well as Fourier-feature PINNs, and constrained Deep Ritz (DR) formulations. In DR approaches we introduce two new neural-network formulations, which implement two classical variational characterizations: a Nehari-constrained Deep Ritz method, in which the height is learned by projecting onto the Nehari/Pokhozhaev manifold, and a Weinstein Deep Ritz method, in which a normalized profile minimizes the Weinstein quotient. The Deep Ritz formulations perform best overall, in particular, the Weinstein variant gives the best error–runtime balance. The neural-network based solvers remain less accurate and eficient than classical Petviashvili iteration in 1D, however, the current study provides a proof of concept: the mesh-free formulation, non-reliance on exact profiles or the related features make them promising in higher dimensions and for more complicated nonlinearities and dispersive operators.

## Contents

1. Introduction 1   
2. Ground-state equation and problem setup 4   
3. PINNs Variations 8   
4. Deep Ritz Variations 26   
5. Conclusions and Discussion 44   
Appendix A. Newton direction and the strong Wolfe conditions 45   
References 46

## 1. Introduction

Ground-state and solitary-wave solutions play a profound role in the study of nonlinear dispersive, or wave-type, partial diferential equations: their profiles and their various norms serve as key elements for understanding wave propagation, expressing thresholds in long-time solutions dynamics (to a given PDE), they are the main players in understanding stability or instability mechanisms, and fundamental in the description of singularity formation such as self-similar collapses and blow-ups. The ground state profile equation, describing the stationary ground state or solitary wave profiles (see derivation for several nonlinear dispersive models in Section 2.1), reduces to a nonlinear elliptic-type diferential equation, whose analytical solutions are often dificult to obtain or inaccessible with current analytical methods, and thus, accurate numerical solution is highly desirable.

In the one-dimensional setting with a single-power nonlinearity, we are interested in ground state solutions of the second-order ODE

$$
\begin{array} { r } { Q ^ { \prime \prime } = b Q - \gamma | Q | ^ { p - 1 } Q , \quad p > 1 , \quad b > 0 , \quad \gamma > 0 , } \end{array}\tag{1}
$$

which are positive, smooth, decaying at infinity solutions $( \mathrm { e . g . }$ , in $H ^ { 1 } ( \mathbb { R } ) )$ , thus, the previous equation can be simplified to

$$
Q ^ { \prime \prime } = b Q - \gamma Q ^ { p } , \qquad Q > 0 , \quad Q ^ { \prime } ( 0 ) = 0 , \quad Q ( \infty ) = 0 .\tag{2}
$$

This equation arises, for example, from standing waves in the nonlinear Schr¨odinger (NLS) equation, traveling solitary waves in the generalized Korteweg–de Vries (gKdV) equation and in the nonlinear Klein–Gordon (NLKG) equation. A key feature of the one-dimensional case is that it admits an explicit exact solution

$$
Q ( x ) = A { \mathrm { ~ s e c h } } ^ { \frac { 2 } { p - 1 } } ( \beta x ) ,
$$

with an amplitude A and inverse width $\beta$ depending on given $b , \gamma ,$ , and $p ,$ see (15). This explicit ground truth makes the 1D case an ideal testing and verification setting for numerical benchmarking.

For the purpose of this paper, we consider integer exponents $p > 1$ , typically studying $p = 3$ (subcritical), $p = 5$ (critical) and $p = 7$ (supercritical) cases.

In our previous work [15], we compared neural-network solvers – PINNs and operator learning approaches such as DeepONets and Fourier Neural Operators – with classical methods such as finitediference with Newton’s iterations and Petviashvili method, for computing these 1D profiles. That study showed that while neural-network methods can approximate solitary-wave profiles, classical Petviashvili based solvers remained substantially more accurate and eficient.

The present paper aims to shrink this gap by introducing, developing and evaluating enhanced neuralnetwork-based algorithms for the ground-state equation (1). This explicitly solvable 1D problem serves as a proof of concept and a controlled benchmark for neural solvers, rather than a setting in which they are expected to replace the Petviashvili iteration. Before advancing to more complicated problems, we, therefore, concisely survey the main methods, introduce new features, and compare them in this setting, in the hope that it will allow future development and comparison in more complicated cases and models. Specifically, we propose three directions of improvement:

‚ Higher-order optimization: replacing or augmenting standard Adam training with Gauss–Newton (GN) and Levenberg–Marquardt (LM) refinement strategies, applied to the PINN residual after an initial Adam warm-up phase;

‚ Fourier-feature PINNs (FF–PINNs): embedding the scalar coordinate input into a space with Fourier basis before passing it to the MLP (multilayer perceptron), in order to reduce the spectral bias of standard coordinate-based networks;

‚ Constrained Deep Ritz formulations: several variational methods that minimize the action functional or related quotients over constrained neural-network trial classes (instead of the profile equation itself), including an action-based constrained Deep Ritz formulation, a Nehari-constrained Deep Ritz formulation, and a Weinstein-quotient-based Deep Ritz formulation.

The last point is worth commenting on further as we develop neural implementations via Deep Ritz (DR) formulations of two classical variational characterizations for obtaining ground state solutions in focusing dispersive PDE: the Nehari-constrained (NDR) and the Weinstein Deep Ritz (WDR).

The WDR formulation is based on minimizing the Weinstein functional quotient (sometimes also called Lyapunov functional or nonlinear Rayleigh, e.g., see [20]) defined as

$$
R _ { b } = \operatorname* { i n f } _ { u \in H ^ { 1 } ( \mathbb { R } ) \backslash \{ 0 \} } \frac { \Vert u ^ { \prime } \Vert _ { L ^ { 2 } } ^ { 2 } + b \Vert u \Vert _ { L ^ { 2 } } ^ { 2 } } { \Vert u \Vert _ { L ^ { p + 1 } } ^ { 2 } } ,\tag{3}
$$

over a normalized, hard-constrained neural-network trial class. One of the main advantages of the WDR formulation, compared to the action-based constrained Deep Ritz method, is that WDR method does not require prescribing the peak amplitude Qp0q from the exact solution (so its training objective can be formed without an explicit profile), thus, making it particularly suitable for problems where no closedform solution is available. The first method, the constrained Deep Ritz, in a general class of solutions may not have a converging result, as it relies on the initial maximum value $A = Q ( 0 )$ , which in its turn helps selecting the correct branch of positive ground state solutions. We first improve that in the Nehari-constrained Deep Ritz (NDR) formulation, by incorporating the first Pokhozhaev (also called Nehari) identity (20) into the formulation and the second Pokhozhaev identity (21) into the loss. We then observe that changing the action functional optimization to the Weinstein quotient (3) optimization further improves the results ( e.g., gives a more favorable error-runtime balance).

Our results show that these enhanced methods substantially improve upon the standard Adam-trained PINN baseline. Among the methods tested, the complete Deep Ritz procedures, including their final residual refinement, give the strongest overall performance, with the new Weinstein Deep Ritz formulation achieving the best (among DR variants) error–runtime for $p = 3$ (cubic) and $p = 5$ (quintic). For the sharper $p \ = \ 7$ (septic) profile, the Weinstein and Nehari-constrained formulations produce errors on the same order, and better compared with the other formulations. The PINN and FF–PINN variants also benefit from higher-order optimization and improved feature representation, but they remain more sensitive to loss formulation, optimizer choice, initialization, and computational device. Overall, although the enhanced neural-network solvers do not yet match the accuracy of Petviashvili-type iterations reported in [15] for the one-dimensional setting, they represent a significant step forward in neural-network-based computation of nonlinear ground states. We expect that in higher-dimensional problems or with more intricate nonlinearities or dispersion, the neural-network-based methods will be an essential addition to the field.

The paper is organized as follows. In Section 2 we set up the ground-state equation and its variational structure, describe the explicit solutions, boundary conditions, domain truncation, the Pokhozhaev identities, and introduce the Weinstein variational characterization used in the Deep Ritz formulations. Section 3 covers PINNs and FF–PINNs, including the use of second-order optimization and comparisons with standard Adam training. In Section 4 we study Deep Ritz formulations, including constrained Deep Ritz, Nehari-constrained Deep Ritz, and Weinstein-type Deep Ritz methods. Section 5 concludes with a discussion of the main findings.

Acknowledgments. We thank Javier G´omez-Serrano for helpful discussions and for suggesting the use of a Fourier-feature-type PINN architecture and higher-order optimization strategies, such as Gauss– Newton, quasi–Newton, and Adam followed by higher-order refinement, as possible ways to improve upon the standard Adam-trained PINN baseline motivated by the strong Petviashvili performance observed in our previous work.

The training on basic neural-network-based approaches for PDE came out of the STEM summer ML seminar that S.R. co-organized in Summer 2025 (with Zhongming Wang), which was partially supported by the NSF grant DUE/EDU-2221491 (PI: S. Roudenko); A.D.R. and C.H. participated in that seminar; A.D.R., C.H. and S.R. were partially supported by the NSF grants DUE/EDU-2221491 and DMS-2452782.

Declaration of usage of Large Language Models. During the preparation of this work the authors used GPT-5.5 and Claude Opus 4.7 in order to assist with code development and debugging. GPT-5.6

Sol and Claude Opus 4.7 were later used for grammar and language editing of the manuscript. While using these tools, the authors reviewed and decided whether the content is relevant, and if it was, it was re-worked and edited before incorporating into the paper; the authors take full responsibility for the content of the article.

## 2. Ground-state equation and problem setup

In this section, we describe the profile equation, its origin from the NLS, gKdV (we write this equation with an absolute value in nonlinearity for consistency of notation), and NLKG models, the explicit solutions, the finite-domain formulation used in the numerical experiments, and the variational structure underlying the Deep Ritz methods.

2.1. Dispersive models and profile reduction. The nonlinear dispersive models considered in this paper are the focusing, power-type one-dimensional equations:

$$
\begin{array} { r } { i u _ { t } + u _ { x x } + \gamma | u | ^ { p - 1 } u = 0 , } \end{array}
$$

$$
\mathrm { ( N L S ) }\tag{4}
$$

$$
u _ { t } + \big ( u _ { x x } + \gamma | u | ^ { p - 1 } u \big ) _ { x } = 0 ,
$$

$$
\mathrm { ( g K d V ) }\tag{5}
$$

$$
u _ { t t } - u _ { x x } + b u - \gamma | u | ^ { p - 1 } u = 0 ,
$$

$$
\mathrm { ( N L K G ) }\tag{6}
$$

with parameters $b > 0 , \gamma > 0$ and integer nonlinearity exponent $p > 1$

The sign convention in front of the nonlinear term is chosen so that these equations are focusing, and hence, possess solitary-wave or ground-state solutions (the defocusing case with the opposite sign does not admit ground states). These models arise in nonlinear optics, plasma physics, and water waves; see, for example, [1, 19, 56, 57, 43, 21, 6] and references therein. The solitary waves or ground states are spatially localized solutions that propagate without changing shape and often play a key role in characterizing the long-time dynamics of the flow, forming the main components of the soliton resolution conjecture [40, 46].

Each of the models (4)–(6) enjoys conservation laws on its maximal interval of existence (see, e.g., the textbooks Cazenave [8], Tao [45], Linares–Ponce [26]). For the NLS equation (4), the conserved mass and Hamiltonian (energy) are given by

$$
M [ u ( t ) ] = \int _ { \mathbb { R } } | u ( t , x ) | ^ { 2 } d x , \qquad E [ u ( t ) ] = \int _ { \mathbb { R } } \Big ( \frac 1 2 | u _ { x } | ^ { 2 } - \frac { \gamma } { p + 1 } | u | ^ { p + 1 } \Big ) d x ,\tag{7}
$$

while for the gKdV equation (5) one has analogous conservation of the L<sup>2</sup>-norm and a suitable Hamiltonian functional. (There is also a momentum in NLS or an integral of the solution in gKdV conservation, but we omit those in this paper.) In the NLKG equation (6), the conserved energy in the phase space $( u , u _ { t } )$ is

$$
\begin{array} { r } { E [ u ( t ) , u _ { t } ( t ) ] = \displaystyle \int _ { \mathbb R } \left( \frac { 1 } { 2 } | u _ { t } | ^ { 2 } + \frac { 1 } { 2 } | u _ { x } | ^ { 2 } + \frac { b } { 2 } | u | ^ { 2 } - \frac { \gamma } { p + 1 } | u | ^ { p + 1 } \right) d x . } \end{array}\tag{8}
$$

Solitary-wave or ground-state solutions are obtained by substituting special ansatzes that reduce the PDE to an ODE for a spatial profile Q. For the NLS equation (4), the standing-wave ansatz $u ( x , t ) =$ $e ^ { i b t } Q ( x )$ yields the profile equation

$$
Q ^ { \prime \prime } - b Q + \gamma | Q | ^ { p - 1 } Q = 0 .\tag{9}
$$

For the gKdV equation (5), the traveling-wave ansatz $u ( x , t ) = Q ( x - b t )$ with speed $b > 0$ similarly gives $Q ^ { \prime \prime } = b Q - \gamma | Q | ^ { p - 1 } Q$ after integration and use of the localization condition $Q ( \xi ) \to 0$ as $| \xi | \to \infty$

For the NLKG equation (6), the traveling-wave ansatz $u ( x , t ) = Q ( x - c t )$ with speed c leads, after rescaling parameters, to the same equation (9).

In all three models, we look for positive $H ^ { 1 }$ solutions of (9), which are known to exist and are unique $( \mathrm { e . g . }$ , see [4] and [22]), therefore, ground-state profiles $Q > 0$ satisfy the second-order profile ODE

$$
Q ^ { \prime \prime } = b Q - \gamma Q ^ { p } .\tag{10}
$$

Under suitable assumptions on p, these ground states can be constructed variationally as minimizers of some functionals such as the action functionals (with or without constraints) or the energy functional at fixed mass and are orbitally stable (in 1D for $p < 5 )$ in the sense of Weinstein [51, 52] and Grillakis– Shatah–Strauss [12, 13], for a review see [20, §2.2]. For gKdV, well-posedness and scattering are discussed in the work of Kenig–Ponce–Vega [17], and the solitary waves and their (in)stability, for example, in Bona–Souganidis–Strauss [7] and references therein

In this paper we focus on the accurate numerical solutions of the profile equation (10) via neural-network approaches enhanced with various optimizers and modified with diferent formulations to achieve more eficient and higher precision results compared to our previous study [15].

2.2. Problem formulation and explicit solutions. We consider localized positive solutions of the one-dimensional nonlinear elliptic equation

$$
Q ^ { \prime \prime } = b Q - \gamma Q ^ { p } ,\tag{11}
$$

$$
Q ^ { \prime } ( 0 ) = 0 , \quad Q ( \infty ) = 0 ,\tag{12}
$$

with parameters $b > 0 , \ \gamma > 0$ and exponent $p > 1$ . As we mentioned in the introduction, for the purpose of this paper, we consider integer exponents, typically, $p = 3 , 5 , 7 .$ The condition $Q ^ { \prime } ( 0 ) = 0$ corresponds to the even symmetry of the full-line ground state, while $Q ( \infty ) = 0$ imposes localization, which is exponential (in the case of a standard Laplacian or second order derivative operator).

Since $Q ( x )  0$ as $x \to \infty$ and $p > 1$ , the nonlinear term in (11) is asymptotically lower order in the tail. Thus, for large x, the decay is governed by the linearized equation

$$
- Q ^ { \prime \prime } + b Q = 0 ,
$$

whose localized solution is proportional to $e ^ { - { \sqrt { b } } x }$ , and thus, satisfies

$$
Q ^ { \prime } ( x ) + \sqrt { b } Q ( x ) = 0 .
$$

Therefore, on the finite half-interval $[ 0 , L ]$ , it is typical in numerical approaches to replace the second condition in (12) by the mixed asymptotic (sometimes called Robin) boundary condition

$$
Q ^ { \prime } ( L ) + \sqrt { b } Q ( L ) = 0 .\tag{13}
$$

We take that into consideration and discuss the boundary conditions in the next subsection.

As we mentioned in the introduction, the ground state profile equation (11) admits an explicit nontrivial solution of the form

$$
Q ( x ) = A \mathrm { s e c h } ^ { \frac { 2 } { p - 1 } } ( \beta x ) \equiv Q _ { \mathrm { e x a c t } } ( x ) ,\tag{14}
$$

where

$$
A = A ( b , \gamma , p ) = \left( \frac { b ( p + 1 ) } { 2 \gamma } \right) ^ { \frac { 1 } { p - 1 } } , \quad \beta = \beta ( b , \gamma , p ) = \frac { \sqrt { b } ( p - 1 ) } { 2 } .\tag{15}
$$

This explicit profile is used both as a reference solution for computing errors and, in some neural-network formulations, as a way to prescribe the correct peak amplitude $Q ( 0 ) = A .$ . (Specifically, in what follows, we use $Q _ { 0 } = A$ as input in Dirichlet PINN, FF-PINN, and action-based constrained DR baselines, whereas in NDR and WDR we determine the amplitude variationally. We also point out that apart from the prescribed boundary value in those baselines, no interior solution samples enter the training losses. The present hyperparameter selection uses errors against the exact cubic profile, which we discuss next.)

The cubic case

$$
Q ^ { \prime \prime } = Q - Q ^ { 3 } ,
$$

corresponding to $b = \gamma = 1$ and $p \ = \ 3 .$ , is used as the main benchmark for hyperparameter tuning. After suitable configurations are selected from the cubic benchmark, the optimized methods are applied typically to the cubic, quintic $( p = 5 )$ , and septic $( p = 7 )$ cases in the main comparisons. It should be mentioned that for higher nonlinearities it might also be possible to optimize the parameters for each nonlinearity independently to improve errors and avoid anomalies that we observe later (e.g., see Figure 5).

2.3. Domain truncation and boundary conditions. In our simulations, the infinite-domain problem is approximated on $[ 0 , L ]$ . We use $L = 3 0$ , which is suficiently large for the benchmark cases considered here; for example, when $b = \gamma = 1$ and $p = 3$ , the exact value $Q ( 3 0 )$ is on the order of 10<sup>´13</sup>. We note that although all our computations are performed on the interval r0, 30s, the solution profiles are plotted only on r0, 10s in order to enlarge the details from the peak-amplitude region, which is typically the most challenging for the neural-network approximation to resolve accurately.

Consider the following finite-domain boundary conditions:

$$
Q ( 0 ) = A , \qquad Q ( L ) = 0 ,\tag{16}
$$

$$
Q ^ { \prime } ( 0 ) = 0 , \qquad Q ( L ) = 0 ,\tag{17}
$$

$$
Q ( 0 ) = A , \qquad Q ^ { \prime } ( L ) + \sqrt { b } Q ( L ) = 0 ,\tag{18}
$$

$$
Q ^ { \prime } ( 0 ) = 0 , \qquad Q ^ { \prime } ( L ) + \sqrt { b } Q ( L ) = 0 .\tag{19}
$$

We refer to (16) as the Dirichlet formulation, (17) as the Neumann–Dirichlet formulation, (18) as the Robin–Dirichlet formulation, and (19) as the Neumann–Robin formulation.

On the truncated interval r0, Ls, the decay condition at infinity may be approximated either by the Dirichlet condition $Q ( L ) = 0$ or by the asymptotic Robin condition $Q ^ { \prime } ( L ) + \sqrt { b } Q ( L ) = 0$ . Since $Q ( x ) =$ $\mathcal { O } ( e ^ { - \sqrt { b } x } )$ as $x \to \infty$ , the Robin condition incorporates the leading-order behavior of the linearized tail and is generally more accurate on short computational domains. In our computations, $b = 1$ and $L \geqslant 1 5$ (typically $L \ = \ 3 0 )$ , so that $Q ( L )$ is already negligibly small relative to the observed neural-network approximation errors. We therefore retain the Dirichlet condition for simplicity (and use either (16) or (17)), while noting that the Robin condition may be advantageous when a shorter interval is used (e.g., $L = 8 )$ or when b is very small. We note that in higher-dimensional settings or diferent nonlinearities, such boundary conditions would need to be taken into account.

2.4. Pokhozhaev identities. For later use in the constrained Deep Ritz formulations in Section 4, we record the two integral identities satisfied by ground state solutions to (11). In 1D, multiplying the equation by $Q$ and integrating by parts gives

$$
\mathcal { P } _ { 1 } ( Q ) : = b \| Q \| _ { L ^ { 2 } } ^ { 2 } + \| \hat { \sigma } _ { x } Q \| _ { L ^ { 2 } } ^ { 2 } - \gamma \| Q \| _ { L ^ { p + 1 } } ^ { p + 1 } = 0 .\tag{20}
$$

The corresponding scaling identity (obtained by multiplying by $x Q ^ { \prime }$ and integrating) is

$$
\mathcal { P } _ { 2 } ( Q ) : = b \| Q \| _ { L ^ { 2 } } ^ { 2 } - \| \partial _ { x } Q \| _ { L ^ { 2 } } ^ { 2 } - \frac { 2 \gamma } { p + 1 } \| Q \| _ { L ^ { p + 1 } } ^ { p + 1 } = 0 .\tag{21}
$$

We use these identities either as penalties in the action-based and Nehari-constrained DR formulations, and as diagnostics for Weinstein DR formulation in Section 4.

For the purpose of the computations on the finite interval $[ 0 , L ]$ , we denote the following quantities by $M = \textstyle \int _ { 0 } ^ { L } Q ^ { 2 } d x , K = \textstyle \int _ { 0 } ^ { L } | Q ^ { \prime } | ^ { 2 } d x$ , and $N = \textstyle \int _ { 0 } ^ { L } | Q | ^ { p + 1 } d x$ , and rewrite the above Pokhozhaev identities, which now contain the boundary terms. Assuming a suficiently smooth solution of the ODE, integration by parts gives the exact finite-interval identities, namely,

$$
P _ { 1 , L } = K + b M - \gamma N + Q ( 0 ) Q ^ { \prime } ( 0 ) - Q ( L ) Q ^ { \prime } ( L ) = 0 ,
$$

and

$$
P _ { 2 , L } = b M - K - \frac { 2 \gamma } { p + 1 } N + L | Q ^ { \prime } ( L ) | ^ { 2 } - b L Q ( L ) ^ { 2 } + \frac { 2 \gamma L } { p + 1 } | Q ( L ) | ^ { p + 1 } = 0 .
$$

2.5. Variational characterization via the Weinstein quotient. The ground-state equation (11) admits a variational characterization that plays a key role in the Weinstein Deep Ritz method developed in Section 4. Recalling the equation (11), for a given $b > 0$ we define the Weinstein functional quotient

$$
R _ { b } : = \operatorname* { i n f } _ { u \in H ^ { 1 } ( \mathbb { R } ) \backslash \{ 0 \} } \frac { \| u ^ { \prime } \| _ { L ^ { 2 } ( \mathbb { R } ) } ^ { 2 } + b \| u \| _ { L ^ { 2 } ( \mathbb { R } ) } ^ { 2 } } { \| u \| _ { L ^ { p + 1 } ( \mathbb { R } ) } ^ { 2 } } .\tag{22}
$$

If v is a positive optimizer of this quotient, normalized by $\| v \| _ { L ^ { p + 1 } ( \mathbb { R } ) } \ = \ 1$ , then $q _ { b } ( v ) \ = \ R _ { b }$ , where $q _ { b } ( u ) : = \| u ^ { \prime } \| _ { L ^ { 2 } } ^ { 2 } + b \| u \| _ { L ^ { 2 } } ^ { 2 }$ , and the Euler–Lagrange equation is

$$
- v ^ { \prime \prime } + b v = R _ { b } v ^ { p } .\tag{23}
$$

The ground-state profile is then recovered by the rescaling

$$
Q = \left( { \frac { R _ { b } } { \gamma } } \right) ^ { \frac { 1 } { p - 1 } } v .\tag{24}
$$

This classical characterization is related (after spatial rescaling) to the sharp interpolation variational principle of Weinstein [50], and is especially convenient for a Deep Ritz implementation. The quotient (22) is invariant under multiplication of u by a nonzero constant (and thus, does not select the trivial solution), and the normalization constraint $\| v \| _ { L ^ { p + 1 } } = 1$ removes the remaining scale invariance. For fixed b and $p ,$ the variational training is independent of $\gamma _ { : }$ which enters only through the final amplitude rescaling (24). Crucially, this method does not require prescribing the peak amplitude $Q ( 0 ) = A$ from the exact solution (14), making it suitable for computing ground states when no closed-form solution is available.

2.6. Related work. Numerical solutions to ODE have been extensively studied in the literature using classical approaches (see discussion in our previous paper [15] on that and references therein). Classical numerical methods, as we showed in [15], typically provide reliable accuracy and convergence for ground state solutions of (10), although usually require mesh design and problem-specific tuning. In contrast, neural-network-based solvers—including physics-informed neural networks (PINNs) [36]—approximate the profile through trainable function classes and can be implemented in a mesh-free manner.

Recent work on PINNs has shown that their accuracy depends strongly on the optimization method and that first-order training followed by higher-order refinement can improve convergence [37]. These findings are directly related to our comparison of Adam [18], L–BFGS [27], and Adam–L–BFGS. PINN training has also been treated as a nonlinear least-squares problem using Gauss-Newton/Levenberg Marquardt (GN/LM) methods, in which damping is used to improve stability [24, 31, 39]. The successful use of first-order training followed by high-precision GN/LM refinement for challenging nonlinear PDE profiles is closely related to, and motivates, our Adam–GN/LM solver [49]. Related work on Fourier features has shown that they help neural networks represent high-frequency and multiscale functions and can reduce spectral bias in PINNs [44, 48], motivating our Fourier-feature PINN (FF–PINN) formulation. For the variational methods, the Deep Ritz approach trains a neural network by minimizing an energy or an action functional rather than a diferential-equation residual [10]. However, direct minimization may not select the desired ground-state branch, unless more conditions or constrains are added. Recent neural approaches that incorporate normalization or projection constraints include [2, 9], and together with recent Deep Ritz work for stationary Schr¨odinger problems in [55], provide additional methods for the constrained neural variational methods considered here. Variational characterizations based, on the other hand, on sharp Gagliardo–Nirenberg inequalities and normalized gradient flows, and yield ap proaches that can properly identify the desired ground states [51, 47, 29, 11]. We consider some of these and suggest novel approaches to construct and compare constrained neural solvers to obtain the ground state solutions in the focusing dispersive PDE, for example, by including Nehari or Pokhozhaev manifolds or the Weinstein quotient. Neural-network implementations of these constrained formulations have recently been developed in [9], who enforce the $L ^ { p + 1 }$ -normalization and Nehari manifold constraints as hard constraints through normalization and projection layers. Their setting is the defocusing action ground state, for which the action is bounded below and an unconstrained formulation is also available. In the focusing problem, which we consider, the action is unbounded below (see Section 4.2), so a constraint is not one option among several but a structural necessity. Our contribution is therefore not the variational characterizations themselves, which are classical, but their realization (including the combined implemen tation and numerical assessment) for the truncated profile equation (10): hard-constrained trial classes that encode symmetry, positivity and the endpoint condition simultaneously, a boundary-corrected finite interval Pokhozhaev penalty, as well as a staged Adam followed by Gauss–Newton/Levenberg–Marquardt residual refinement. A recent work [28] analyzes the $L ^ { p + 1 }$ -normalized gradient flow with asymptotic La grange multiplier for computing the action ground state of the focusing NLS, using a gradient-flow rather than a neural-network approach.

2.7. Computational setup. All reported runtimes in the tables are recorded as elapsed wall-clock times and measured using time.perf counter(). Therefore, the reported CPU times should be interpreted as practical implementation-dependent runtimes on the stated hardware, rather than hardware-independent performance measurements. The computations were performed on a device equipped with a 64-bit Windows laptop with an AMD Ryzen 5 7535HS CPU at 3.30 GHz, 6 cores / 12 threads, 16 GB RAM (15.3 GB usable), integrated Radeon graphics, and a dedicated NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB VRAM<sup>1</sup>. In Sections 3.1.3, 3.2.1, and 4.1.1, for comparison purposes we also use a “secondary” device equipped with an AMD Ryzen 7 9700X (3.8 GHz base, 5.5 GHz boost, 8 cores / 16 threads), 32 GB of RAM (31.1 GB usable), and a dedicated NVIDIA GeForce RTX 4070 Super GPU with 12 GB of VRAM.

For CUDA-enabled experiments, the reported elapsed times reflect the end-to-end runtime of the implemented PyTorch computation on this device, including CPU-side launching and synchronization overheads as well as GPU computation. We therefore label all secondary-device timings as GPU+CPU rather than GPU-only: the CUDA device carries the tensor and neural-network computations, but the optimizer control logic, backtracking decisions, checkpointing, logging, data transfers, and any calls to CPU-based linear algebra or NumPy routines remain on the CPU. The reported wall-clock time is thus the combined execution of GPU-accelerated tensor operations and CPU-side back-end work, and should not be read as a measure of processor performance alone. Accordingly, the primary- and secondary-device results should be interpreted as a comparison between the two complete workstation configurations rather than as an isolated comparison of CPU and GPU performance. This convention applies to Sections 3.1.3, 3.2.1, 4.1.1, 4.2.1, and 4.3.1.

All reported computations use double precision (float64) and fixed np.random.seed(42) for reproducibility. The tabulated values correspond to the individual runs reported in the respective experiments.

## 3. PINNs Variations

We first study residual-based neural-network solvers for (11). The common benchmark problem, bound ary conditions, finite-domain truncation, and timing protocol were specified in Section 2. In this section, we focus on Physics-Informed Neural Networks (PINNs) and Fourier-Feature PINNs (FF–PINNs), together with the optimization strategies used to train them.

For each method, we first describe the neural-network ansatz and loss function. We then perform a hyperparameter study on the cubic trial problem

$$
Q ^ { \prime \prime } = Q - Q ^ { 3 } .
$$

Then we use the selected configuration for analysis and comparison of the cubic, quintic and septic cases. The purpose of this section is to assess the accuracy, robustness, and computational cost of residual-based neural-network solvers, including with the enhancements, before comparing them with the variational Deep Ritz formulations in Section 4.

Building on our previous study of neural-network-based solvers for solitary wave computation [15], which considered PINNs, DeepONets, and FNOs, we extend the numerical comparison in this section by incorporating higher-order optimization strategies and Fourier-feature PINNs.

The goal of this section is to evaluate the numerical performance in terms of accuracy and computation eficiency, where computation cost is measured using wall-clock elapsed time time.perf counter(). These baselines results later compared against the results obtained from the Deep Ritz formulations of Section 4.

3.1. Physics-Informed Neural Network (PINN). In the PINN framework [36], we approximate the solitary-wave profile $Q ( x )$ by the standard feedforward multilayer perceptron (MLP) $Q _ { \theta } ( x )$ with m hidden layers and n neurons in each layer, namely,

$$
Q ( x ) \approx Q _ { \theta } ( x ) ,
$$

where θ denotes the collection of trainable network parameters. All spatial derivatives appearing in the loss are evaluated using the reverse-mode automatic-diferentiation engine autograd provided by PyTorch [34, Sec. 4.3]. Rather than approximating derivatives by finite diferences, autograd records the elementary tensor operations executed during the forward evaluation of $Q _ { \theta } ( x )$ and applies the chain rule through the resulting computational graph, so that the computed derivatives are exact up to floating point error [3, Secs. 1 and 3]. The spatial derivatives $Q _ { \theta } ^ { \prime } ( x )$ and $Q _ { \theta } ^ { \prime \prime } ( x )$ are obtained by diferentiating the network output with respect to the input x; computing $Q _ { \theta } ^ { \prime \prime } ( x )$ requires retaining the graph of the first derivative so that $Q _ { \theta } ^ { \prime } ( x )$ can itself be diferentiated. The same construction subsequently allows the PINN loss, which depends on $Q _ { \theta }$ and its spatial derivatives, to be diferentiated with respect to the trainable parameters θ.

In the numerical experiments, we test several smooth activation functions, including tanh, SiLU, Gaussian, super-Gaussian, sech, and the nonstandard activation

$$
\sigma _ { \mathrm { S i L U P L U S } } ( x ) = \frac { x } { 1 + e ^ { - x ^ { 2 } } } .\tag{25}
$$

No single activation function is assumed to be optimal across all neural-network formulations. Instead, the activation is treated as a hyperparameter and selected separately for each method, boundary-condition formulation, and optimizer configuration. The activation chosen for each final setup is reported together with the remaining selected hyperparameters in the corresponding configuration tables.

For the single-power stationary problem, the network minimizes the residual associated with the finitedomain profile ODE (10). Specifically, we define

$$
\begin{array} { r } { \mathcal { R } ( x ; \theta ) = Q _ { \theta } ^ { \prime \prime } ( x ) - b Q _ { \theta } ( x ) + \gamma Q _ { \theta } ^ { p } ( x ) . } \end{array}\tag{26}
$$

We evaluate this residual at collocation points $\{ x _ { i } \} _ { i = 1 } ^ { N _ { f } } \subset [ 0 , L ]$ , where the computational domain is r0, Ls. The boundary points $x = 0$ and $x = L$ are treated separately via the Dirichlet boundary loss

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { B C } } ( \theta ) = | Q _ { \theta } ( 0 ) - Q _ { 0 } | ^ { 2 } + | Q _ { \theta } ( L ) | ^ { 2 } . } \end{array}\tag{27}
$$

Here, the residuals are scalar, $\mathrm { s o } \ | \cdot |$ denotes the usual absolute value. We write the total PINN loss as

$$
\mathcal { L } _ { \mathrm { P I N N } } ( \theta ) = \frac { 1 } { N _ { f } } \sum _ { i = 1 } ^ { N _ { f } } \lvert \mathcal { R } ( x _ { i } ; \theta ) \rvert ^ { 2 } + \lambda _ { \mathrm { B C } } \left( \lvert Q _ { \theta } ( 0 ) - Q _ { 0 } \rvert ^ { 2 } + \lvert Q _ { \theta } ( L ) \rvert ^ { 2 } \right) ,\tag{28}
$$

where $\lambda _ { \mathrm { B C } }$ is the boundary-weight parameter. The network parameters are then determined by solving

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \mathcal { L } _ { \mathrm { P I N N } } ( \theta ) .\tag{29}
$$

Building on the PINN results that we obtained in [15], our present work focuses specifically on higherorder optimization strategies for training the PINN loss. In particular, we investigate the Gauss–Newton/ Levenberg–Marquardt (GN/LM) method and the quasi-Newton L–BFGS method. We also consider hybrid procedures, in which Adam is used only as an initialization or warm-up stage before refinement with GN/LM or L–BFGS. The general PINN algorithm with second-order optimizers is summarized in Algorithm 1 below.

3.1.1. Higher-order optimizer refinements for PINNs. The PINN loss in (28) has a nonlinear least-squares structure, that is, we can rewrite it as

$$
\mathcal { L } ( \boldsymbol { \theta } ) = \| \boldsymbol { F } ( \boldsymbol { \theta } ) \| _ { l ^ { 2 } } ^ { 2 } ,\tag{30}
$$

where $F ( \theta )$ is the finite-dimensional vector of rescaled residuals,

$$
\begin{array} { r } { F ( \theta ) = \left[ \begin{array} { c } { N _ { f } ^ { - 1 / 2 } \mathcal { R } ( x _ { 1 } ; \theta ) } \\ { \vdots } \\ { N _ { f } ^ { - 1 / 2 } \mathcal { R } ( x _ { N _ { f } } ; \theta ) } \\ { \sqrt { \lambda _ { \mathrm { B C } } } \big ( Q _ { \theta } ( 0 ) - Q _ { 0 } \big ) } \\ { \sqrt { \lambda _ { \mathrm { B C } } } Q _ { \theta } ( L ) } \end{array} \right] . } \end{array}\tag{31}
$$

Thus, the first $N _ { f }$ components of $F ( \theta )$ are the interior ODE residuals evaluated at the collocation points and scaled by $N _ { f } ^ { - 1 / 2 }$ , while the final two components are the scaled Dirichlet boundary residuals. With this notation, $\begin{array} { r } { \| \dot { F } ( \boldsymbol { \theta } ) \| _ { l ^ { 2 } } ^ { 2 } = \sum _ { i = 1 } ^ { N _ { f } + 2 } | F _ { j } ( \boldsymbol { \theta } ) | ^ { 2 } } \end{array}$ , so (28) and $( 3 0 ) \mathrm { - ( 3 1 ) }$ describe the same loss.

Since Fpθq depends on $Q _ { \theta }$ and its spatial derivatives, the parameter gradient of the loss must be propagated through the operations that construct $Q _ { \theta } ^ { \prime } ( x )$ and $Q _ { \theta } ^ { \prime \prime } ( x )$

$$
\begin{array} { r } { \theta \longmapsto Q _ { \theta } ( x ) \longmapsto ( Q _ { \theta } ^ { \prime } ( x ) , Q _ { \theta } ^ { \prime \prime } ( x ) ) \longmapsto F ( \theta ) \longmapsto \mathcal { L } ( \theta ) . } \end{array}
$$

The gradient therefore carries the dependence of both the network values and their spatial deriva tives on the weights and biases; this nested diferentiation is handled by the same PyTorch automaticdiferentiation graph described above [3, 34]. It is this propagation through the spatial derivatives that makes the residual Jacobian $J _ { F }$ below more expensive to form and, in general, more poorly conditioned than in an ordinary regression problem (a point we return to when introducing Levenberg–Marquardt damping in (37)).

Writing $\theta \in \mathbb { R } ^ { N _ { \theta } }$ , where $N _ { \theta }$ is the total number of trainable weights and biases, let

$$
J _ { F } ( \theta ) = \frac { \partial F ( \theta ) } { \partial \theta } \in \mathbb { R } ^ { ( N _ { f } + 2 ) \times N _ { \theta } }
$$

denote the Jacobian of the residual vector with respect to the trainable parameters. The chain rule applied to (30) gives

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { L } ( \boldsymbol { \theta } ) = 2 J _ { F } ( \boldsymbol { \theta } ) ^ { T } F ( \boldsymbol { \theta } ) = 2 \sum _ { i = 1 } ^ { N _ { f } + 2 } F _ { i } ( \boldsymbol { \theta } ) \nabla _ { \boldsymbol { \theta } } F _ { i } ( \boldsymbol { \theta } ) ,\tag{32}
$$

where the sum over residual components is exactly the matrix product $J _ { F } ( \theta ) ^ { T } F ( \theta )$

This least-squares formulation motivates comparing optimizers that use diferent levels of derivative information. The three families considered here use it diferently: Adam is a first-order adaptive-gradient method that uses $\nabla _ { \boldsymbol { \theta } } \mathcal { L }$ together with component-wise running estimates formed from present and past gradients; L–BFGS is a quasi-Newton method that infers curvature of the scalar loss from changes in successive parameter gradients; and GN/LM is a nonlinear least-squares method that uses the residual vector $F ( \theta )$ directly to build a problem-specific curvature approximation. Throughout, “higher-order” is used broadly for methods that incorporate curvature information, or an approximation to it, rather than in the sense of forming exact second derivatives: L–BFGS estimates curvature from gradient diferences, and GN/LM builds its curvature approximation from the residual Jacobian.

We use Adam as the baseline warm-up method, since it is a first-order adaptive-gradient method based on running estimates of the first and second moments of the gradient [18]. (Here, the “second moment” is a running average of squared gradient components, not a second derivative of the loss.) Adam is useful during the initial stage of training because it is relatively robust from random initialization and tolerates poor gradient scaling; however, as a first-order method, it may converge slowly once the iterate is already near a local minimizer. We therefore test two curvature-informed refinement strategies: L–BFGS and damped Gauss–Newton / Levenberg–Marquardt (GN/LM).

‚ L–BFGS refinement. L–BFGS is a limited-memory quasi-Newton method for large-scale optimization [27, Sec. 2], [33, Secs. 6.1 and 7.2]. Rather than forming the $N _ { \theta } \times N _ { \theta }$ Hessian $\nabla _ { \theta } ^ { 2 } \mathcal { L }$ , which is impractical to store for large networks, it maintains an implicit approximation $H _ { k }$ to the inverse Hessian built from a limited number of recent parameter and gradient diferences, and produces search directions

$$
p _ { k } = - H _ { k } \nabla _ { \theta } \mathcal { L } ( \theta _ { k } ) .\tag{33}
$$

The parameters are then updated by $\theta _ { k + 1 } = \theta _ { k } + \alpha _ { k } p _ { k }$ , where the step length $\alpha _ { k } > 0$ is selected by a strong Wolfe line search [53, 54, 32]. The strong Wolfe conditions enforce both a suficient decrease in the loss and a curvature condition that keeps the BFGS inverse-Hessian approximation positive definite, and hence the search direction a descent direction; we use the strong Wolfe line search provided by the PyTorch implementation of L–BFGS. The derivation of (33) from the Newton direction and the precise Wolfe conditions are standard and are collected in Appendix A. The stability of the strong Wolfe search is particularly helpful for the FF–PINNs of Section 3.2, where the Fourier-feature embedding makes the optimization landscape more sensitive to step size. Unlike GN/LM, however, L–BFGS is a general purpose method for smooth scalar objectives and does not exploit the decomposition of the loss into the individual residuals in $F ( \theta )$

‚ Gauss–Newton/Levenberg–Marquardt refinement. The GN/LM method is especially natural here because the PINN loss has the nonlinear least-squares form (30). Linearizing F near the current iterate $\theta _ { k }$

$$
\begin{array} { r } { F ( \theta _ { k } + \delta ) \approx F ( \theta _ { k } ) + J _ { F } ( \theta _ { k } ) \delta , } \end{array}\tag{34}
$$

which follows by stacking the first-order component expansions $F _ { i } ( \theta _ { k } + \delta ) \approx F _ { i } ( \theta _ { k } ) + \nabla _ { \theta } F _ { i } ( \theta _ { k } ) ^ { T } \delta$ , whose gradient rows $\nabla _ { \boldsymbol { \theta } } F _ { i } ( \boldsymbol { \theta } _ { k } ) ^ { T }$ are exactly the rows of $J _ { F } ( \theta _ { k } )$ . Gauss–Newton chooses the increment $\delta _ { k }$ by minimizing the linearized residual norm $\| F ( \theta _ { k } ) + J _ { F } ( \theta _ { k } ) \delta \| _ { l ^ { 2 } } ^ { 2 }$ , giving the normal equations

$$
\begin{array} { r } { J _ { F } ( \theta _ { k } ) ^ { T } J _ { F } ( \theta _ { k } ) \delta _ { k } = - J _ { F } ( \theta _ { k } ) ^ { T } F ( \theta _ { k } ) , \qquad \theta _ { k + 1 } = \theta _ { k } + \delta _ { k } . } \end{array}\tag{35}
$$

Equivalently, $J _ { F } ( \theta _ { k } ) ^ { T } \big ( F ( \theta _ { k } ) + J _ { F } ( \theta _ { k } ) \delta _ { k } \big ) = 0 ,$ , which states that the remaining linearized residual is orthogonal to the column space of $J _ { F } ( \theta _ { k } )$ . Geometrically, $J _ { F } ( \theta _ { k } ) \delta _ { k }$ is chosen to approximate $- F ( \theta _ { k } )$ as closely as the first-order model allows: if exact cancellation is attainable the linearized residual is zero, and otherwise the step yields the smallest attainable $l ^ { 2 } \mathrm { - n o r m }$ . Since the components of $F ( \theta )$ are the ODE and boundary residuals, $F ( \theta ) = 0$ would mean the network satisfies the equation and boundary conditions exactly at all sampled points; thus zero is the ideal, and Gauss–Newton seeks the best locally attainable reduction toward it.

The connection to the Hessian follows by diferentiating (32):

$$
\nabla _ { \boldsymbol { \theta } } ^ { 2 } \mathcal { L } ( \boldsymbol { \theta } ) = 2 J _ { F } ( \boldsymbol { \theta } ) ^ { T } J _ { F } ( \boldsymbol { \theta } ) + 2 \sum _ { i = 1 } ^ { N _ { f } + 2 } F _ { i } ( \boldsymbol { \theta } ) \nabla _ { \boldsymbol { \theta } } ^ { 2 } F _ { i } ( \boldsymbol { \theta } ) .\tag{36}
$$

Gauss–Newton neglects the second term and uses $2 J _ { F } ( \theta ) ^ { T } J _ { F } ( \theta )$ as a curvature approximation, which is especially appropriate near a solution where the residual components $F _ { i } ( \theta )$ are small. Unlike L–BFGS, which infers curvature from successive loss gradients, Gauss–Newton builds its approximation directly from the residual Jacobian.

In practice we use the damped Levenberg–Marquardt modification

$$
\left( J _ { F } ( \theta _ { k } ) ^ { T } J _ { F } ( \theta _ { k } ) + \lambda I \right) \delta _ { k } = - J _ { F } ( \theta _ { k } ) ^ { T } F ( \theta _ { k } ) , \qquad \lambda > 0 ,\tag{37}
$$

in which λI regularizes the linear system and improves stability when $J _ { F } ^ { T } J _ { F }$ is ill-conditioned or the linearization (34) is inaccurate. For large λ the increment approaches $\begin{array} { r } { \delta _ { k } \approx - \frac { 1 } { \lambda } J _ { F } ( \theta _ { k } ) ^ { T } F ( \theta _ { k } ) } \end{array}$ , a scaled negative-gradient step by (32), while as $\lambda  0$ the update approaches the Gauss–Newton step; the undamped method is recovered at λ “ 0 [24, 31]. This damping is particularly relevant for PINN training because, as noted above, the residual depends on derivatives of the network output and the residual Jacobian can be poorly conditioned.

Algorithm 1 PINN Method with Second-Order Optimizers for $Q ^ { \prime \prime } = b Q - \gamma Q ^ { p }$   
1 Choose domain r0, Ls, collocation points $\left\{ x _ { j } \right\} _ { j = 1 } ^ { N _ { f } }$ , and boundary condition type.   
2 Define a neural network $q _ { \theta } ( x )$   
3 Define the PINN approximation by   
$Q _ { \theta } ( x ) = q _ { \theta } ( x ) .$   
4 Define the ODE residual   
$\mathcal { R } _ { \theta } ( x ) = Q _ { \theta } ^ { \prime \prime } ( x ) - b Q _ { \theta } ( x ) + \gamma Q _ { \theta } ( x ) ^ { p } .$   
5 Define   
$\mathcal { L } _ { \mathrm { O D E } } ( \theta ) = \frac { 1 } { N _ { f } } \sum _ { j = 1 } ^ { N _ { f } } \left| \mathcal { R } _ { \theta } ( x _ { j } ) \right| ^ { 2 } .$   
6 Define the boundary loss   
$\mathcal { L } _ { \mathrm { B C } } ( \theta ) = \left| Q _ { \theta } ( 0 ) - Q _ { 0 } \right| ^ { 2 } + \left| Q _ { \theta } ( L ) \right| ^ { 2 }$   
7 Define the total loss   
L pθq “ L pθq \` λ L pθq.   
8 Choose one of the optimizer modes   
Opt P tAdam, L–BFGS, GN/LM, Adam Ñ L–BFGS, Adam Ñ GN/LMu.   
9 for $n = 1 , 2 , \ldots , N _ { \mathrm { s t e p s } }$ do   
10 Use automatic diferentiation to compute $Q _ { \theta } ^ { \prime }$ and $Q _ { \theta } ^ { \prime \prime } .$   
11 Compute ${ \mathcal { L } } _ { \mathrm { O D E } } ( \theta ) , { \mathcal { L } } _ { \mathrm { B C } } ( \theta ) .$ , and ${ \mathcal { L } } _ { \mathrm { P I N N } } ( \theta )$   
12 Update the trainable parameters θ according to the selected optimizer mode Opt.   
13 Periodically resample the random subset of the collocation points and store checkpoint diagnostics.   
14 end for   
15 Return the trained approximation $Q _ { \theta } ( x )$

In summary, Adam uses first-order gradients with component-wise adaptive scaling but does not mode curvature coupling between parameter directions; L–BFGS constructs an implicit inverse-Hessian ap proximation from parameter and gradient diferences; and GN/LM exploits the nonlinear least-squares structure to form the problem-specific curvature approximation $J _ { F } ( \theta ) ^ { T } J _ { F } ( \theta )$ directly from the residual Jacobian.

In addition to applying L–BFGS and GN/LM directly from the network initialization, we consider hybrid strategies in which an Adam warm-up precedes L–BFGS or GN/LM refinement. Such hybrid procedures are common in PINN practice and have been reported to improve accuracy and/or training eficiency relative to a single optimizer; see, e.g., [30, 38, 37]. Recent work indicates that stronger secondorder optimizers can substantially outperform the classical Adam–L–BFGS procedure in demanding high accuracy regimes [16, 49]. When L–BFGS or GN/LM is applied directly from the network initialization, without an Adam warm-up, we refer to the resulting procedure as pure L–BFGS or pure GN/LM, respectively.

3.1.2. Implementation of PINN with second order optimizers. In our implementation in the following sections, we record specific parameters and errors, mostly described in tables, such as Table 1. Throughout the numerical results, we report the standard relative $L ^ { 2 }$ error and $L ^ { \infty }$ error, namely,

$$
\frac { \lVert Q _ { \theta } - Q _ { \mathrm { e x a c t } } \rVert _ { L ^ { 2 } ( 0 , L ) } } { \lVert Q _ { \mathrm { e x a c t } } \rVert _ { L ^ { 2 } ( 0 , L ) } } \quad \mathrm { a n d } \quad \lVert Q _ { \theta } - Q _ { \mathrm { e x a c t } } \rVert _ { L ^ { \infty } ( 0 , L ) } .
$$

For the higher-order optimizer PINN comparison, the selected L–BFGS, Adam–L–BFGS, GN/LM, and Adam–GN/LM configurations from Tables 2 and 5 were evaluated with the Dirichlet boundary condition (16) for the ODE $Q ^ { \prime \prime } = Q - Q ^ { p }$ , with $p = 3 , 5$ , and 7. The results in Tables 3 and 6 show that all four refinement strategies are capable of reducing the training loss $\mathcal { L } ( \boldsymbol { \theta } )$ to small values, but the resulting profile accuracy depends strongly on the optimizer family. We use the term training loss here because $\mathcal { L } ( \boldsymbol { \theta } )$ is the residual-based loss minimized by the PINN, whether the network is trained directly with Adam or further refinement using L–BFGS or GN/LM. More generally, training objective is used when referring to both the PINN losses and the variational objectives minimized in the Deep Ritz methods as seen in Section 4.

<table><tr><td>Parameter</td><td>Values tested / description</td></tr><tr><td>Domain length L</td><td>30</td></tr><tr><td>Pure L-BFGS runs</td><td>L-BFGS initialized directly from the network initialization</td></tr><tr><td>Adam-L-BFGS runs</td><td>Adam warm-up followed by L-BFGS refinement</td></tr><tr><td>Activation functions</td><td> $e ^ { - x ^ { 2 } } , e ^ { - x ^ { 4 } }$  tanh(x), SiLU(x), SiLUPLUS(x), , and sech(x)</td></tr><tr><td>Number of hidden layers</td><td> $3 , 4 , \mathrm { o r ~ 5 ~ }$ </td></tr><tr><td>Neurons per layer</td><td>32, 64, or 96</td></tr><tr><td>Number of collocation points  $N _ { f }$ </td><td> $2 ^ { 8 } , 2 ^ { 9 } , 2 ^ { 1 0 } , \mathrm { o r ~ } 2 ^ { 1 1 }$ </td></tr><tr><td>Adam warm-up length</td><td>2,500, 5,000, 7,500, 10,000, 12,500, and 15,000 epochs</td></tr><tr><td>Adam learning rate lr</td><td> $1 0 ^ { - 3 }$  initial; StepLR decay 0.1, 1, 5, or 25</td></tr><tr><td>Boundary condition weight λBC L-BFGS refinement steps</td><td>25, 50, 75, 100, 125, 150, or 200</td></tr><tr><td>L-BFGS history size</td><td>20, 50, or 100</td></tr><tr><td>L-BFGS max iterations per step</td><td>1, 5, 10, 20, or 30</td></tr><tr><td>L-BFGS learning rate</td><td>0.5 and 1.0</td></tr><tr><td>Selection criterion</td><td>Smallest  $L ^ { 2 }$  error, with runtime used to break near-ties</td></tr></table>

Table 1. Schedule quantities varied in the L–BFGS and Adam–L–BFGS optimizer tests.

For these higher-order PINN refinement methods, hyperparameter selection was performed using the representative cubic problem $Q ^ { \prime \prime } = Q - Q ^ { 3 }$ . We fixed $p = 3$ and swept through the PINN, L–BFGS, and GN/LM optimizer parameters summarized in Tables 1 and 4. After selecting the best configurations according to the smallest $L ^ { 2 }$ error, with runtime used to break near-ties, the same optimizer settings were then held fixed and evaluated on the cases $p = 3 , 5$ , and 7. Although the cubic case was used during hyperparameter selection, the final selected configuration was rerun for $p = 3$ on the primary computational device so that all three reported benchmark cases were obtained using the same finalized setup and timing environment. We remark that when no exact profile is available, hyperparameters could instead be selected using the ODE and boundary residuals on an independent, finer validation grid, together with Pokhozhaev identities (or one of them, which is not algebraically enforced by the trial function). Then properties such as positivity, localization, centering of the computed profile under domain and quadrature refinement could also be checked. In this paper, we do not address this, and wil consider this elsewhere, e.g., see an example of equations with combined potentials in [14].

The internal L–BFGS settings were fixed throughout our computations: strong-Wolfe line search, gradient tolerance $1 0 ^ { - 1 0 }$ , and change tolerance $1 0 ^ { - 1 2 }$ . These lower-level internal settings were not swept. The optimizer-schedule comparison varied the optimizer type, activation function, network architecture, number of collocation points, Adam warm-up length, and L–BFGS refinement schedule. For the checkpoint based Adam–L–BFGS tests, Adam checkpoints were compared at diferent warm-up lengths, followed by L–BFGS refinement checkpoints. The selected configurations are therefore optimizer-dependent rather than common to all L–BFGS-based runs. For the pure L–BFGS case, the selected Dirichlet configuration used 50 L–BFGS steps, history size 100, a maximum of 10 inner iterations per step, L–BFGS learning rate 1.0, and $\lambda _ { \mathrm { B C } } = 5$ . For the Adam–L–BFGS case, the selected Dirichlet configuration used a 15,000-step Adam warm-up followed by 25 L–BFGS refinement steps, history size 100, and a maximum of 5 inner iteration per step, L–BFGS learning rate 1.0, and $\lambda _ { \mathrm { B C } } = 1$ . In both cases, the strong-Wolfe line search was used for the L–BFGS updates. Table 1 summarizes the L–BFGS schedule quantities considered in the optimizer tests, while Table 2 reports the selected Dirichlet configurations for each L–BFGS-based optimizer.

<table><tr><td>Optimizer</td><td>Activation</td><td>Hidden Layer</td><td>Neurons per Layer</td><td> $N _ { f }$ </td><td>Adam Warm-up</td><td>L-BFGS Steps</td><td>History size</td><td>Max Iters per Step</td><td>L-BFGS LR</td><td>λBC</td></tr><tr><td>L-BFGS</td><td>tanh(x)</td><td>4</td><td>32</td><td> $2 ^ { 1 0 }$ </td><td></td><td>50</td><td>100</td><td>10</td><td>1.0</td><td>5.0</td></tr><tr><td>Adam-L-BFGS</td><td>tanh(x)</td><td>4</td><td>64</td><td> $2 ^ { 9 }$ </td><td>15,000</td><td>25</td><td>100</td><td>5</td><td>1.0</td><td>1.0</td></tr></table>

Table 2. Selected L–BFGS and Adam–L–BFGS for Dirichlet boundary conditions.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>PINN with L-BFGS optimization</td><td rowspan=1 colspan=4>PINN with Adam-L-BFGS optimization</td></tr><tr><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1> ${ \mathcal { L } } _ { \mathrm { P I N N } } ( \theta )$ </td><td rowspan=1 colspan=1>CPU time</td><td rowspan=1 colspan=1> $L ^ { \infty }$ error</td><td rowspan=1 colspan=1> $\overline { { L ^ { 2 } } }$ error</td><td rowspan=1 colspan=1> ${ \mathcal { L } } _ { \mathrm { P I N N } } ( \theta )$ </td><td rowspan=1 colspan=1>CPU time</td><td rowspan=1 colspan=1> $L ^ { \infty }$ error</td><td rowspan=1 colspan=1> $\overline { { L ^ { 2 } } }$ error</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1.314525e-08</td><td rowspan=1 colspan=1>15.328056</td><td rowspan=1 colspan=1>7.162973e-03</td><td rowspan=1 colspan=1>5.864162e-03</td><td rowspan=1 colspan=1>5.917457e-08</td><td rowspan=1 colspan=1>353.818283</td><td rowspan=1 colspan=1>7.896724e-03</td><td rowspan=1 colspan=1>6.463914e-03</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1.679169e-07</td><td rowspan=1 colspan=1>14.411380</td><td rowspan=1 colspan=1>8.026411e-03</td><td rowspan=1 colspan=1>7.032712e-03</td><td rowspan=1 colspan=1>2.705050e-07</td><td rowspan=1 colspan=1>338.015580</td><td rowspan=1 colspan=1>5.668667e-03</td><td rowspan=1 colspan=1>4.915231e-03</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>9.080274e-08</td><td rowspan=1 colspan=1>10.815290</td><td rowspan=1 colspan=1>2.407888e-03</td><td rowspan=1 colspan=1>2.187603e-03</td><td rowspan=1 colspan=1>2.613929e-07</td><td rowspan=1 colspan=1>326.965276</td><td rowspan=1 colspan=1>2.374574e-03</td><td rowspan=1 colspan=1>2.225127e-03</td></tr></table>

Table 3. Comparison of PINN performance with L–BFGS and Adam–L–BFGS optimization and Dirichlet boundary conditions (16) for $Q ^ { \prime \prime } = Q - Q ^ { p }$

In Table 3, the left group of columns reports the pure L–BFGS results, while the right group reports the Adam–L–BFGS results. For the L–BFGS-based methods, the pure L–BFGS runs are substantially cheaper than Adam–L–BFGS, and the added Adam warm-up does not uniformly improve the final accuracy. For $p = 3$ , the pure L–BFGS run gives the smaller $L ^ { \infty }$ and $L ^ { 2 }$ errors despite requiring much less CPU time.

For $p = 5$ , Adam–L–BFGS gives the smaller $L ^ { \infty }$ and $L ^ { 2 }$ errors. For $p = 7 .$ , Adam–L–BFGS gives a slightly smaller $L ^ { \infty }$ error, whereas pure L–BFGS gives a slightly smaller $L ^ { 2 }$ error, with both methods producing errors of the same order. We therefore find that the Adam warm-up does not uniformly improve the L–BFGS refinement stage, even though it can help for the $p = 5$ case.

Figure 1 shows in the left panel that pure L–BFGS rapidly decrease the PINN training loss during the initial refinement steps before reaching a plateau. The profile plot in the right panel shows the learned profiles converging toward the exact ground state given by (14) as the number of L–BFGS steps increases. We observe that the final profile after 50 L–BFGS steps agrees closely with the exact profile, up to the errors reported in the left group of columns in Table 3. For $p \ : = \ : 5$ , the final $L ^ { \infty }$ and $L ^ { 2 }$ errors are $8 . 0 2 6 4 1 1 \times 1 0 ^ { - 3 }$ and $7 . 0 3 2 7 1 2 \times 1 0 ^ { - 3 }$ , respectively, and are therefore both of order $1 0 ^ { - 3 }$ . We show the computations for the representative quintic case in Figure 1, as it is the $L ^ { 2 } { \mathrm { - c r i t i c a l } }$ (borderline) case in one dimension, with similar convergence behavior observed for the cubic and septic cases.

![](images/d0cde70a8dbe2a64a1b8ae3a1b7f2a8ce37d3d242397a96b991d3c81f0683820.jpg)

![](images/5459cbc9b16c73991ed321721e129b4054c3dd7e9a59df8adefffba79e04b271.jpg)

Figure 1. Training loss (left) and ground state solution profile convergence (right) for L–BFGS PINN (with parameters as seen in Table 2) for $Q ^ { \prime \prime } = Q - Q ^ { 5 }$ . The (black dashed) curve labeled Exact in the right panel is the explicit solution given by (14).  
![](images/1661fa4ec037e9a80bebcc7e32f3feed52be0b90b7607f63168c0990982092c9.jpg)

![](images/b03c32c0d232e25e12c89e834c04c7834a2d79af9fe66b42b626e96908852613.jpg)  
Figure 2. Training loss (left) and ground state solution profile convergence (right) for Adam–L–BFGS PINN (with parameters as seen in Table 2) for $Q ^ { \prime \prime } = Q - Q ^ { 5 }$ . The (black dashed) curve labeled Exact in the right panel is the explicit solution given by (14). The final learned profile corresponds to 15,000 Adam steps followed by 25 L–BFGS steps.

Figure 2 shows in the left panel that the Adam warm-up gradually reduces the training loss before the L–BFGS refinement stage produces the final decrease. In the profile plot in the right panel of the same figure, we observe that the learned profiles improve across the recorded Adam and L–BFGS checkpoints and approach the exact ground state given by (14). The labels of the form $1 5 , 0 0 0 + j$ denote the total optimizer-step count, with the first 15,000 epochs corresponding to Adam and the subsequent $j$ steps corresponding to the L–BFGS refinement. The final profile after the 15,000-epoch Adam warm-up and 25 L–BFGS steps agrees closely with the exact profile, up to the errors reported in the right group of columns of Table 3. For $p = 5$ , the final $L ^ { \infty }$ and $L ^ { 2 }$ errors are 5.668667 $\times \ 1 0 ^ { - 3 }$ and $4 . 9 1 5 2 3 1 \times 1 0 ^ { - 3 }$ respectively, and are therefore both of order $1 0 ^ { - 3 }$ . Similar qualitative behavior was also observed for $p = 3$ and $p = 7$

For the GN/LM PINN refinements, the optimizer comparison varied the optimizer type, activation function, network architecture, number of collocation points, Adam warm-up length, GN/LM refinement length, and damping parameters. For the pure GN/LM case, the method was initialized directly from the random network initialization, while in the Adam–GN/LM case, the GN/LM refinement was applied after an Adam warm-up. The damping-related quantities, including the initial damping, maximum number of backtracking steps, damping shrink factor, damping growth factor, and minimum damping, were swept as part of the optimizer tuning.

Parameter Values tested / description   
Domain length L 30   
Pure GN/LM runs GN/LM initialized directly from the network initialization   
Adam–GN/LM runs Adam warm-up followed by $\mathrm { G N / L M }$ refinement   
Activation functions tanhpxq, SiLUpxq, SiLUPLUSpxq, $e ^ { - x ^ { 2 } } , e ^ { - x ^ { 4 } }$ , and sechpxq   
Number of hidden layers 3, 4, or 5   
Neurons per layer 32, 64, or 96   
Number of collocation points $N _ { f }$ $2 ^ { 8 } , 2 ^ { 9 } , 2 ^ { 1 0 } , \mathrm { o r ~ } 2 ^ { 1 1 }$   
Adam warm-up checkpoints 2,500, 5,000, 7,500, 10,000, 12,500, and 15,000 epochs   
Adam learning rate lr $1 0 ^ { - 3 }$ initial; StepLR decay   
Boundary condition weight λ<sub>BC</sub> 0.1, 1. 5 or 25   
GN/LM refinement steps 25, 50, 75, 100, 125, 150, or 200   
Initial damping $5 \times 1 0 ^ { - 2 } , 1 0 ^ { - 3 } ,$ , and $5 \times 1 0 ^ { - 4 }$   
Maximum backtracking steps 2, 4, 6, 8, and 10   
Damping shrink factor Values in r0.4, 1.1s   
Damping growth factor $5 , 1 0 ,$ and 15   
Minimum damping $1 0 ^ { - 1 2 } , 1 0 ^ { - 1 4 } ,$ , and $1 0 ^ { - 1 6 }$   
Selection criterion Smallest $L ^ { 2 }$ error, with runtime time used to break near-ties  
Table 4. Schedule and damping quantities varied in the GN/LM and Adam–GN/LM optimizer tests.

The selected GN/LM-based Dirichlet configurations shared the collocation size, refinement length, damping parameters, and boundary-condition weight, but not the activation function. In both cases, the selected runs used $N _ { f } = 2 ^ { 9 }$ , 150 GN/LM refinement steps, initial damping $1 0 ^ { - 3 }$ , 2 maximum backtracking steps, damping shrink factor 0.6, damping growth factor 10, and $\lambda _ { \mathrm { B C } } = 5$ . Pure GN/LM was started directly from the random network initialization and used sechpxq activation with 4 hidden layers with 64 neurons per layer. Adam–GN/LM used tanhpxq activation with with 5 hidden layers and 32 neurons per layer, following a 15,000-epoch Adam warm-up. Table 4 summarizes the tuning ranges, and Table 5 reports the selected configurations.
<table><tr><td>Optimizer</td><td>Activation</td><td>Hidden Layer</td><td>Neurons per Layer</td><td> $N _ { f }$ </td><td>Adam Warm-up</td><td>GN/LM Steps</td><td>Max Backtracks</td><td>Damping Shrink</td><td>Damping Growth</td><td>Initial Damping</td><td>λBC</td></tr><tr><td>GN/LM</td><td>sech(x)</td><td>4</td><td>64</td><td> $2 ^ { 9 }$ </td><td></td><td>150</td><td>2</td><td>0.6</td><td>10</td><td> $1 0 ^ { - 3 }$ </td><td>5</td></tr><tr><td>Adam-GN/LM</td><td>tanh(x)</td><td>5</td><td>32</td><td> $2 ^ { 9 }$ </td><td>15,000</td><td>150</td><td>2</td><td>0.6</td><td>10</td><td> $1 0 ^ { - 3 }$ </td><td>5</td></tr></table>

Table 5. Selected GN/LM and Adam–GN/LM configurations for Dirichlet boundary conditions.

The GN/LM-based methods give smaller errors than the L–BFGS-based methods in most cases, but at a much higher computational cost when run on the primary CPU device. In particular, pure GN/LM obtains errors near $1 0 ^ { - 4 } – 1 0 ^ { - 5 }$ for $p = 3$ and $p = 5$ , while Adam–GN/LM gives the best result among the PINN optimizers for $p = 7$ . The corresponding convergence plots in Figures 3 and 4 show a rapid reduction of the training objective under GN/LM refinement and close agreement between the computed profile, that the neural network learned, and the exact ground state given by (14). Overall, we find that these results suggest GN/LM refinement is the most accurate PINN refinement mechanism tested here, while the Adam warm-up can improve robustness for the more dificult higher-power case.

In Table 6, the left group of columns reports the pure GN/LM results, while the right group reports the Adam–GN/LM results. We observe that the two methods give errors of comparable order for $p = 3$ while pure GN/LM gives the more accurate profile for $p = 5$ . In particular, the pure GN/LM errors for the quintic case are of order $1 0 ^ { - 5 } .$ , compared with order 10<sup>´4</sup> for Adam–GN/LM. For $p = 7 ,$ , however, the Adam warm-up produces the smaller errors. Thus, we find that, although pure GN/LM attains the smallest error for the cubic and quintic cases, Adam–GN/LM is more robust across the increasing nonlinear test cases.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>PINN with GN/LM optimization</td><td rowspan=1 colspan=4>PINN with Adam-GN/LM optimization</td></tr><tr><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1> ${ \mathcal { L } } _ { \mathrm { P I N N } } ( \theta )$ </td><td rowspan=1 colspan=1>CPU time</td><td rowspan=1 colspan=1> $L ^ { \infty }$ error</td><td rowspan=1 colspan=1> $\overline { { L ^ { 2 } } }$ error</td><td rowspan=1 colspan=1>LPINN(θ)</td><td rowspan=1 colspan=1>CPU time</td><td rowspan=1 colspan=1> $L ^ { \infty }$ error</td><td rowspan=1 colspan=1> $\overline { { L ^ { 2 } } }$ error</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2.763789e-11</td><td rowspan=1 colspan=1>4532.584975</td><td rowspan=1 colspan=1>1.193728e-04</td><td rowspan=1 colspan=1>9.808560e-05</td><td rowspan=1 colspan=1>6.763931e-13</td><td rowspan=1 colspan=1>1639.024311</td><td rowspan=1 colspan=1>1.417711e-04</td><td rowspan=1 colspan=1>1.158408e-04</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1.015142e-10</td><td rowspan=1 colspan=1>4940.055658</td><td rowspan=1 colspan=1>6.839973e-05</td><td rowspan=1 colspan=1>5.982481e-05</td><td rowspan=1 colspan=1>1.328934e-10</td><td rowspan=1 colspan=1>1872.362646</td><td rowspan=1 colspan=1>5.831789e-04</td><td rowspan=1 colspan=1>5.065382e-04</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>3.184690e-08</td><td rowspan=1 colspan=1>5205.814115</td><td rowspan=1 colspan=1>2.681668e-03</td><td rowspan=1 colspan=1>2.388390e-03</td><td rowspan=1 colspan=1>4.723684e-10</td><td rowspan=1 colspan=1>1812.308855</td><td rowspan=1 colspan=1>9.558302e-04</td><td rowspan=1 colspan=1>8.583382e-04</td></tr></table>

Table 6. Comparison of PINN performance with GN/LM and Adam–GN/LM optimization and Dirichlet boundary conditions (16) for $Q ^ { \prime \prime } = Q - Q ^ { p }$

![](images/e912d55de1eed90775e2a53bdb58a4a9ae39924d4153ab01054671a83870ed30.jpg)

![](images/562a3ddc93d482f07a89f1d3c8073c7c0871a7202eaa237afb8a82d932323f3c.jpg)  
Figure 3. Training loss (left) and ground state solution profile convergence (right) for GN/LM PINN (with parameters as seen in Table 5) for $Q ^ { \prime \prime } = Q - Q ^ { 5 }$ . The (black dashed) curve labeled Exact in the right panel is the explicit solution given by (14).

Figure 3 shows in the left panel that pure GN/LM produces a sharp and sustained decrease in the PINN training loss over the refinement steps. In the right panel, we observe that the successive learned profiles approach the exact ground state given by (14). By step 50, the learned and exact curves are already visually indistinguishable at the plotting scale. We nevertheless continue the optimization to the selected final value of 150 GN/LM steps and report the errors at that final step in the left group of columns of Table 6 because the order of error at step 50 is only $1 0 ^ { - 3 }$ , compared with $1 0 ^ { - 5 }$ at step 150. For $p = 5$ , the final $L ^ { \infty }$ and $L ^ { 2 }$ errors are $6 . 8 3 9 9 7 3 \times 1 0 ^ { - 5 }$ and $5 . 9 8 2 4 8 1 \times 1 0 ^ { - 5 }$ , respectively, and are therefore both of order $1 0 ^ { - 5 }$ . The error at step 50 is not reported because this curve is an intermediate visualization checkpoint rather than the final trained configuration used in the comparison table, although the secondary-device results in Section 3.1.3 show that pure GN/LM can be sensitive to damping and branch selection.

Figure 4 shows that Adam–GN/LM combines the gradual stabilization provided by Adam with a rapid final decrease in the training loss during the GN/LM refinement. In the right panel, we compare the profiles at the recorded Adam and GN/LM checkpoints with the exact ground state given by (14). The labels of the form 15000\`j denote the total optimizer-step count, with the first 15,000 steps corresponding to the Adam warm-up and the subsequent j steps corresponding to GN/LM refinement. We retain only a subset of the available checkpoints in the legend, with no more than eight displayed curves, so that the plot remains readable. Thus, the displayed refinement-step values need not be equally spaced. We observe that the final profile after 150 GN/LM refinement steps agrees closely with the exact profile, up to the errors reported in the right group of columns of Table 6. For $p \ : = \ : 5 .$ , the final $L ^ { \infty }$ and $L ^ { 2 }$ errors are $5 . 8 3 1 7 8 9 \times 1 0 ^ { - 4 }$ and $5 . 0 6 5 3 8 2 \times 1 0 ^ { - 4 }$ , respectively, thus, both are of order 10<sup>´4</sup>, supporting Adam–GN/LM as the most robust PINN refinement strategy (among the tested methods). Although pure GN/LM gives the smaller error for the quintic case, we regard Adam–GN/LM as more robust among the cases tested, since Adam warm-up provides a more reliable initialization for the GN/LM refinement stage.

![](images/dc9327e135d15469ceb7e0029a3d0d610fe449704d62a0c888fad252375af731.jpg)

![](images/7688be4e6a9ef5c136bcaacd4a8a57d15d029566608e0d57d5f87fffff260360.jpg)  
Figure 4. Training loss (left) and ground state solution profile convergence (right) for Adam–GN/LM PINN (with parameters as seen in Table 5) for $Q ^ { \prime \prime } = Q - Q ^ { 5 }$ . The (black dashed) curve labeled Exact in the right panel is the explicit solution given by (14). The final learned profile corresponds to 15,000 Adam steps followed by 150 GN/LM steps.

3.1.3. Comparison with the “secondary” computational device for PINNs with higher-order optimizer refinements. As a secondary-device check, we repeated the selected higher-order PINN refinement experiments on the separate GPU-capable workstation described in Section 2.7. The purpose of this comparison is not to introduce a new hyperparameter sweep, but to evaluate the same selected PINN configurations under a diferent hardware and linear algebra environment. Therefore, the architecture, collocation size, boundary-condition weight, optimizer schedule, and stopping criteria were kept fixed, while only the computational device was changed. The larger-damping GN/LM experiments considered below are reported separately as a sensitivity test and are not part of this fixed-configuration device comparison. As in the preceding subsection, we may refer to L–BFGS or GN/LM applied directly from the network initialization, without an Adam warm-up, as pure L–BFGS or pure GN/LM, respectively.

As explained in Section 2.7, these timings are reported as GPU+CPU rather than GPU-only, and are not directly comparable to CPU-only timings as a measure of processor performance alone.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>PINN with L-BFGS optimization</td><td rowspan=1 colspan=4>PINN with Adam-L-BFGS optimization</td></tr><tr><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1> $\overline { { \mathcal { L } _ { \mathrm { P I N N } } ( \theta ) } }$ </td><td rowspan=1 colspan=1>GPU+CPU time</td><td rowspan=1 colspan=1> $\overline { { L ^ { \infty } } }$ error</td><td rowspan=1 colspan=1> $\overline { { L ^ { 2 } } }$ error</td><td rowspan=1 colspan=1> ${ \mathcal { L } } _ { \mathrm { P I N N } } ( \theta )$ </td><td rowspan=1 colspan=1>GPU+CPU time</td><td rowspan=1 colspan=1> $\overline { { L ^ { \infty } } }$ error</td><td rowspan=1 colspan=1> $\overline { { L ^ { 2 } } }$ error</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>5.329101e-08</td><td rowspan=1 colspan=1>7.505031</td><td rowspan=1 colspan=1>6.583645e-03</td><td rowspan=1 colspan=1>5.421850e-03</td><td rowspan=1 colspan=1>3.593178e-08</td><td rowspan=1 colspan=1>62.361742</td><td rowspan=1 colspan=1>3.780430e-03</td><td rowspan=1 colspan=1>3.079454e-03</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>7.202707e-08</td><td rowspan=1 colspan=1>6.157369</td><td rowspan=1 colspan=1>5.845897e-03</td><td rowspan=1 colspan=1>5.090068e-03</td><td rowspan=1 colspan=1>1.164522e-07</td><td rowspan=1 colspan=1>63.238091</td><td rowspan=1 colspan=1>2.992726e-03</td><td rowspan=1 colspan=1>2.691600e-03</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>4.067500e-08</td><td rowspan=1 colspan=1>6.788750</td><td rowspan=1 colspan=1>1.609846e-03</td><td rowspan=1 colspan=1>1.493517e-03</td><td rowspan=1 colspan=1>2.353134e-07</td><td rowspan=1 colspan=1>61.959399</td><td rowspan=1 colspan=1>9.146505e-04</td><td rowspan=1 colspan=1>1.052700e-03</td></tr></table>

Table 7. Secondary-device GPU+CPU comparison of PINN performance with L–BFGS and Adam–L–BFGS optimization and Dirichlet boundary conditions (16) for $Q ^ { \prime \prime } = Q { - } Q ^ { p } .$

Tables 7 and 8 show that the secondary GPU+CPU runs substantially reduce the reported wall-clock time for the higher-order refinement experiments. This reduction is especially clear for the Adam–L– BFGS and GN/LM-based methods, where the secondary device gives much shorter run times than the corresponding primary-device CPU runs. The L–BFGS-based methods remain broadly comparable to the primary-device CPU runs, with errors of the same general order. The Adam–GN/LM runs are also stable across the secondary-device comparison and remain close to the exact ground state given by (14) for $p = 3 , 5$ , and 7.

In Table 7, the left group of columns reports the pure L–BFGS results, while the right group reports the Adam–L–BFGS results. We observe that all of the profile errors are of order $1 0 ^ { - 3 }$ . On the secondary device, the Adam warm-up reduces the profile errors for each value of $p ,$ but it also increases the GPU+CPU time from $6 - 8$ seconds for pure L–BFGS to approximately $6 2 - 6 3$ seconds for Adam–L– BFGS. Thus, we find the same general trade-of observed on the primary device: the Adam warm-up can improve the final profile, but at a substantially greater computational cost.

The main discrepancy in the secondary-device PINN runs occurs for the pure GN/LM optimizer. With the selected initial damping used in Table 5, the septic case $p = 7$ does not converge to the centered ground-state branch. Instead (although the profile preserves the prescribed value $Q ( 0 ) )$ , the GN/LM iterates develop an additional bump further to the right. This produces the large $L ^ { \infty }$ and $L ^ { 2 }$ errors reported for the $p = 7$ entry, even though the training loss is still reduced. Hence, we have to distinguish between reduction of the sampled residual-based loss and convergence to the desired ground-state profile.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>PINN with GN/LM optimization</td><td rowspan=1 colspan=4>PINN with Adam-GN/LM optimization</td></tr><tr><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1> $\overline { { \mathcal { L } _ { \mathrm { P I N N } } ( \theta ) } }$ </td><td rowspan=1 colspan=1>GPU+CPU time</td><td rowspan=1 colspan=1> $\overline { { L ^ { \infty } } }$ error</td><td rowspan=1 colspan=1> $\overline { { L ^ { 2 } } }$ error</td><td rowspan=1 colspan=1> $\overline { { \mathcal { L } _ { \mathrm { P I N N } } ( \theta ) } }$ </td><td rowspan=1 colspan=1>GPU+CPU time</td><td rowspan=1 colspan=1> $\overline { { L ^ { \infty } } }$ error</td><td rowspan=1 colspan=1> $\overline { { L ^ { 2 } } }$ error</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4.025347e-11</td><td rowspan=1 colspan=1>538.109242</td><td rowspan=1 colspan=1>9.699502e-05</td><td rowspan=1 colspan=1>7.902533e-05</td><td rowspan=1 colspan=1>1.294946e-12</td><td rowspan=1 colspan=1>506.161066</td><td rowspan=1 colspan=1>5.594483e-05</td><td rowspan=1 colspan=1>4.555836e-05</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>4.330355e-11</td><td rowspan=1 colspan=1>534.144825</td><td rowspan=1 colspan=1>2.525764e-04</td><td rowspan=1 colspan=1>2.192186e-04</td><td rowspan=1 colspan=1>1.701413e-10</td><td rowspan=1 colspan=1>511.624151</td><td rowspan=1 colspan=1>5.432745e-04</td><td rowspan=1 colspan=1>4.717845e-04</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>5.115845e-06</td><td rowspan=1 colspan=1>538.628306</td><td rowspan=1 colspan=1>1.256511</td><td rowspan=1 colspan=1>1.411610</td><td rowspan=1 colspan=1>1.485399e-10</td><td rowspan=1 colspan=1>531.554408</td><td rowspan=1 colspan=1>6.169997e-04</td><td rowspan=1 colspan=1>5.539982e-04</td></tr></table>

Table 8. Secondary-device GPU+CPU comparison of PINN performance with GN/LM and Adam–GN/LM optimization and Dirichlet boundary conditions (16) for $Q ^ { \prime \prime } = Q - Q ^ { p }$

In Table 8, the left group of columns reports the pure GN/LM results, while the right group reports the Adam–GN/LM results. For $p = 3$ , both methods give errors between orders of $1 0 ^ { - 4 }$ to $1 0 ^ { - 5 }$ , while for $p = 5$ both give errors of order $1 0 ^ { - 4 }$ . The most significant diference occurs for $p = 7 \colon$ pure GN/LM profile has errors of order one because it converges to the incorrect branch, whereas Adam–GN/LM gives $L ^ { \infty }$ and $L ^ { 2 }$ errors of $6 . 1 6 9 9 9 7 \times 1 0 ^ { - 4 }$ and $5 . 5 3 9 9 8 2 \times 1 0 ^ { - 4 }$ , respectively, both of order $1 0 ^ { - 4 }$ . Thus, one can note that the Adam warm-up prevents the branch-selection failure in the septic case without increasing the secondary-device run time relative to pure GN/LM.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>PINN with GN/LM optimization</td></tr><tr><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1> ${ \mathcal { L } } _ { \mathrm { P I N N } } ( \theta )$ </td><td rowspan=1 colspan=1>GPU+CPU time</td><td rowspan=1 colspan=1> $L ^ { \infty }$ error</td><td rowspan=1 colspan=1> $L ^ { 2 }$ error</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2.483710e-08</td><td rowspan=1 colspan=1>591.697312</td><td rowspan=1 colspan=1>1.413589e+00</td><td rowspan=1 colspan=1>1.414096e+00</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1.058034e-06</td><td rowspan=1 colspan=1>586.388381</td><td rowspan=1 colspan=1>1.318001e+00</td><td rowspan=1 colspan=1>1.417144e+00</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>1.221674e-09</td><td rowspan=1 colspan=1>574.305437</td><td rowspan=1 colspan=1>6.810766e-04</td><td rowspan=1 colspan=1>6.118432e-04</td></tr></table>

Table 9. Secondary-device GPU+CPU comparison of PINN performance with GN/LM with larger initial damping (10<sup>´1</sup>) and Dirichlet boundary conditions (16) for $Q ^ { \prime \prime } = Q { - } Q ^ { p }$

To test whether this failure was caused by insuficient damping, we repeated the pure GN/LM runs with larger initial damping values $1 0 ^ { - 2 }$ and $1 0 ^ { - 1 }$ . Table 9 reports the results obtained with initial damping

10<sup>´1</sup>. This change corrects the septic case and yields an accurate $p = 7$ profile. We note, however, that the same damping choice destabilizes the lower-power cases. For $p = 3$ the learned solution develops a secondary bump away from the origin, while for $p = 5$ the optimizer converges to a negative, of-centered profile rather than the positive, centered at $x = 0$ ground state. We show representative non-ground state profiles in Figure 5. We note that increasing the damping resolves the branch-selection issue for the septic (supercritical) case but introduces the same type of non-ground state convergence for the cubic and quintic cases.

From Table 9, we observe that the larger damping reverses the behavior seen with the selected damping. The cubic and quintic profiles now have $L ^ { \infty }$ and $L ^ { 2 }$ errors of order one, while the septic profile has errors of order $1 0 ^ { - 4 }$ . Thus, no single one of these two initial damping choices reliably selects the desired branch for all three powers when pure GN/LM is initialized directly from the random network parameters.

A plausible explanation is that the pure GN/LM PINN loss does not uniquely select the ground-state branch. We interpret the observed failure as evidence of the branch-selection ambiguity. In the Dirichlet setting, the loss enforces the residual together with $Q ( 0 ) = Q _ { 0 }$ and $Q ( L ) = 0$ , but it does not directly impose positivity, monotone decay, or $Q ^ { \prime } ( 0 ) = 0$ . As a result, the network can satisfy the boundary conditions and maintain a small residual at the sampled collocation points while developing a secondary interior bump, producing a non-ground-state profile.

This issue is more pronounced for pure GN/LM because it starts from the random initialization and takes large linearization-based correct steps in a nonconvex least-squares landscape. Small numerical diferences between the devices or linear algebra back-ends may therefore alter the accepted GN/LM steps and steer the iteration toward a diferent local minimizer. Thus, the secondary device double-bump profile does not necessarily indicate a change in the equation or code; rather, we interpret it as showing that pure GN/LM is branch-sensitive unless the desired ground-state structure is enforced or reached through a more stable path such as Adam–GN/LM. In the left panel of Figure 5, we compare the pure

![](images/63f9788aeac09c09ae485bed41095172065fe5a465bffa74f022d2b206f331c9.jpg)

![](images/9440948f2198217c4036af78c9f6afb0a7b6d19fff8dbc4da64088c641b5b8a1.jpg)  
Figure 5. Secondary device ground state solution profile convergence for GN/LM PINN with $p \ : = \ : 7$ with parameters as seen in Table 5 (left) and $p \ : = \ : 5$ with GN/LM initial damping $1 0 ^ { - 1 }$ (right) for $Q ^ { \prime \prime } = Q - Q ^ { p }$ . The (black dashed) curve labeled Exact in the right panel is the explicit solution given by (14).

GN/LM profile for $p = 7 .$ , obtained using the selected damping from Table 4, with the exact ground state given by (14). The final learned profile develops an additional positive bump and does not align with the exact profile; the corresponding $L ^ { \infty }$ and $L ^ { 2 }$ errors in the left group of columns of Table 8 are 1.256511 and 1.411610, respectively, and are therefore of order one. In the right panel, we compare the $p = 5$ profile obtained with the larger initial damping 10<sup>´1</sup> with the exact solution from (14). The final learned solution profile is negative and of-centered rather than positive and centered at the origin, and its $L ^ { \infty }$ and $L ^ { 2 }$ errors from Table 9 are 1.318001 and 1.417144, respectively, again of order one.

In conclusion, we find that pure GN/LM is highly sensitive to damping, initialization, and devicedependent numerical details. We observe that this is indeed the case when testing higher order powers in the nonlinear term, where changing the initial damping yields results similar to those in Figure 5. In contrast, Adam–GN/LM is more robust because the Adam warm-up moves the network closer to the desired ground-state branch before the GN/LM refinement begins. We emphasize that our secondarydevice comparison supports the main conclusion that Adam–GN/LM is the more reliable PINN refinement strategy, while pure GN/LM can achieve high accuracy only when the damping and initialization place the iteration on the correct solution branch.

3.2. Fourier-Feature Physics-Informed Neural Network (FF–PINN). We also test a Fourier feature physics-informed neural network (FF–PINN), in which the scalar coordinate input is first embed ded into a trigonometric feature space before being passed to the multilayer perceptron. In our implementation, the physical coordinate is scaled by the computational length L. When the raw coordinate is included, the embedding is

$$
x \mapsto \left[ { \frac { x } { L } } , \sin \left( 2 \pi { \frac { x } { L } } B \right) , \cos \left( 2 \pi { \frac { x } { L } } B \right) \right] ,
$$

where $B \in \mathbb { R } ^ { d \times N _ { \mathrm { F F } } }$ is a frequency matrix in d dimensions. In our 1D case, B should be referred to as a frequency vector as we are studying the one-dimensional ground state problem.

We also test the corresponding embedding without the raw coordinate,

$$
x \mapsto \left[ \sin \left( 2 \pi { \frac { x } { L } } B \right) , \cos \left( 2 \pi { \frac { x } { L } } B \right) \right] .
$$

The entries of B are initialized from a Gaussian distribution whose scale is determined by $\sigma _ { \mathrm { F F } }$ . We consider both fixed-frequency embeddings, where B is kept fixed after initialization and trainable-frequency embeddings, where B is optimized together with the neural-network weights and biases. This construc tion is related to random Fourier features for kernel approximation [35] and to Fourier-feature embeddings used to reduce the spectral bias of coordinate-based neural networks [44]. Fourier and multiscale feature embeddings have also been studied in the PINN setting for problems where standard PINNs may struggle to represent high-frequency or multiscale components [48]. Thus, inclusion of the raw coordinate $x / L$ is treated as a hyperparameter in the FF–PINN study.

The role of this embedding is to provide the network with oscillatory building blocks before training the MLP. With fixed random frequencies, one obtains a richer input representation without introducing additional trainable frequency parameters. More generally, one could instead treat B as trainable, so that the optimization adjusts the frequencies themselves rather than only the network weights. This may be advantageous when the relevant frequency scale is not known in advance, since the model can adapt its embedding to the problem. However, it also makes the optimization more delicate, because the network must then learn both the shape of the solution and the frequencies used to represent it. In particular, trainable frequencies may drift toward unnecessarily large values, which can produce oscillatory artifacts and reduce stability.

This issue is especially important because the residual contains the second derivative $Q ^ { \prime \prime }$ . Indeed, a high-frequency component of the form sinpkxq satisfies

$$
{ \frac { d ^ { 2 } } { d x ^ { 2 } } } \sin ( k x ) = - k ^ { 2 } \sin ( k x ) ,
$$

so high frequencies are amplified in the diferential operator through the factor $k ^ { 2 }$ . Consequently, overly large Fourier frequencies can have a disproportionate efect on the residual, making the optimization more sensitive and potentially favoring oscillatory low-residual profiles that do not correspond to the desired ground state. For this reason, the Fourier-feature scale and the number of features must be chosen with care.

The motivation for testing this architecture is that our target equation is a stationary profile problem, i.e., $Q ^ { \prime \prime } \ = \ b Q \ - \gamma Q ^ { p }$ , and Fourier-based classical methods, such as Petviashvili iterations combined with FFT discretizations, perform very efectively for localized solitary-wave profiles. The FF–PINN is therefore used to test whether enriching the input representation with sinusoidal features improves the PINN approximation relative to the standard coordinate-input architecture. At the same time, this advantage is not automatic in the present setting: the desired ground state is smooth, positive, centered, and localized, rather than highly oscillatory. Thus, Fourier features increase the expressive power of the network, but they can also introduce unnecessary oscillations.

For this reason, we pair the FF–PINN experiments with ground-state-oriented constraints incorporated directly into the neural network trial function. To target the nontrivial ground-state branch, we use the structured trial function

$$
Q _ { \theta } ( x ) = Q _ { 0 } \frac { \left( 1 - \left( \frac { x } { L } \right) ^ { 2 } \right) ^ { 3 } } { 1 + x ^ { 2 } \mathrm { s o f t p l u s } ( q _ { \theta } ( x ) ) } , \quad Q _ { 0 } = A ( b , \gamma , p ) > 0 .\tag{38}
$$

This type of hard-constraint trial function is consistent with PINN approaches that encode boundary behavior directly in the network ansatz rather than enforcing all constraints only through penalty terms [42, 5]. This ansatz (38) has three useful features. First, the factor

$$
\left( 1 - \left( { \frac { x } { L } } \right) ^ { 2 } \right) ^ { n }
$$

enforces $Q _ { \theta } ( L ) = 0$ , also its derivative vanishes at $x = 0$ . In particular, we take $n = 3$ because near $x = L$ this factor behaves like a constant multiple of $( L { - } x ) ^ { 3 }$ , so the envelope and its first two derivatives vanish at the right boundary. Since the residual involves the second derivative $\boldsymbol { Q } _ { \boldsymbol { \theta } } ^ { \prime \prime } ,$ this stronger flattening reduces boundary-induced curvature and leads to more stable and accurate training. In our experiments, the choice $n = 2$ also satisfied the boundary conditions, but produced less favorable numerical performance. Second, $Q _ { 0 }$ is the fixed peak amplitude from (15), not a trainable parameter. Thus, the reported FF–PINN experiments use exact peak information, a freely trained positive amplitude would require a separate nonzero normalization to rule out collapse toward the trivial solution. Third, the denominator

$$
1 + x ^ { 2 } \mathrm { s o f t p l u s } ( q _ { \theta } ( x ) )
$$

is positive and has zero derivative at $x = 0$ , so it preserves the condition $Q _ { \theta } ^ { \prime } ( 0 ) = 0$ . Additionally, the use of softplus lets the network reshape the profile in the interior regions of the domain, so the ansatz is not just one fixed polynomial envelope. We note that since the hard-constrained ansatz satisfies $Q _ { \theta } ( 0 ) = Q _ { 0 }$ and $Q _ { \theta } ( L ) = 0$ identically, no separate boundary loss or boundary-penalty parameter is required. All FF–PINN results reported below use this hard-constrained formulation.

In this way, the ansatz (38) biases the optimization toward a positive, nontrivial ground-state profile rather than the trivial solution. In particular, the residual and boundary losses alone may not select the positive ground-state branch, especially when the Fourier embedding gives the network additional oscillatory flexibility. Thus, we use a hard ground-state trial function to bias the admissible profiles toward positive, localized, boundary-compatible solutions. This is consistent with the broader PINN literature on enforcing constraints through the architecture or trial function rather than relying only on soft penalty terms, see, e.g., [23, 42]. Recent hard-constrained Fourier-feature PINN formulations provide further motivation for combining Fourier-type embeddings with built-in boundary or structural constraints, as in [25, 41]. We also tested the simpler linear Dirichlet hard ansatz

$$
Q _ { \theta } ( x ) = Q _ { 0 } \left( 1 - \frac { x } { L } \right) + \frac { x } { L } \left( 1 - \frac { x } { L } \right) q _ { \theta } \left( \frac { x } { L } \right) ;
$$

however, the hard-constrained decay ansatz (38) performed better for FF–PINN and was used for its final comparisons. The action-based DR baseline instead uses the linear trial (41), as specified in Section 4.1.

We note that the FF–PINN should be viewed not as a replacement for the classical Fourier spectral method, but as an architectural modification of the PINN: the diferential-equation residual and boundary conditions remain the training objective, while the neural network is supplied with a richer trigonometric input representation. Its performance is therefore assessed by the same diagnostics used for the standard PINN, including the $L ^ { 2 }$ error and the $L ^ { \infty }$ error.

We summarize the implementation of the FF–PINN used in our numerical experiments in Algorithm 2.

Algorithm 2 Fourier-feature PINN with Second-Order Optimizer for $Q ^ { \prime \prime } = b Q - \gamma Q ^ { p }$   
1 Choose domain $[ 0 , L ]$ , collocation points $\{ x _ { j } \} _ { j = 1 } ^ { N _ { f } } .$ , Fourier-feature dimension $N _ { \mathrm { F F } }$ , Fourier scale $\sigma _ { \mathrm { F F } } ,$ and whether to include the   
raw coordinate   
2 Initialize a fixed frequency matrix $B \in \mathbb { R } ^ { 1 \times N _ { \mathrm { F F } } }$ with entries drawn from a Gaussian distribution with scale $\sigma _ { F F } .$   
3 Choose whether B is kept fixed after initialization or trained together with the neural-network parameters.   
4 Define the scaled coordinate   
$\xi = { \frac { x } { L } } .$   
5 Define the Fourier-feature mapping by   
$\Phi ( x ) = \left\{ \begin{array} { l } { { \left[ \xi , \sin ( 2 \pi \xi B ) , \cos ( 2 \pi \xi B ) \right] } } \\ { { } } \\ { { \left[ \sin ( 2 \pi \xi B ) , \cos ( 2 \pi \xi B ) \right] , } } \end{array} \right.$ , if the raw coordinate is included,   
otherwise.   
6 Define a neural network   
$q _ { \theta } ( x ) = \mathrm { M L P } _ { \theta } ( \Phi ( x ) ) .$   
7 Define the FF–PINN approximation by   
$Q _ { \theta } ( x ) = Q _ { 0 } \frac { \left( 1 - \left( \frac { x } { L } \right) ^ { 2 } \right) ^ { 3 } } { 1 + x ^ { 2 } \mathrm { s o f t p l u s } ( q _ { \theta } ( x ) ) } ,$   
where $Q _ { 0 } = A ( b , \gamma , p )$ from (15) is fixed throughout training.   
8 Define the ODE residual   
$\mathcal { R } _ { \theta } ( x ) = Q _ { \theta } ^ { \prime \prime } ( x ) - b Q _ { \theta } ( x ) + \gamma Q _ { \theta } ( x ) ^ { p } .$   
9 Define the residual loss   
$\mathcal { L } _ { \mathrm { O D E } } ( \theta ) = \frac { 1 } { N _ { f } } \sum _ { j = 1 } ^ { N _ { f } } | \mathcal { R } _ { \theta } ( x _ { j } ) | ^ { 2 } .$   
10 Define the total loss   
$\mathcal { L } _ { \mathrm { F F P I N N } } ( \theta ) = \mathcal { L } _ { \mathrm { O D E } } ( \theta ) .$   
11 Choose one of the optimizer modes   
Opt P tAdam, L–BFGS, GN/LM, Adam Ñ L–BFGS, Adam Ñ GN/LM u.   
12 for $n = 1 , 2 , \ldots , N _ { \mathrm { s t e p s } }$ do   
13 Use automatic diferentiation to compute $Q _ { \theta } ^ { \prime }$ and $Q _ { \theta } ^ { \prime \prime } .$   
14 Compute $\mathcal { L } _ { \mathrm { F F P I N N } } ( \theta ) = \mathcal { L } _ { \mathrm { O D E } } ( \theta )$   
15 Update the parameters using the selected optimizer mode $\mathrm { O p t . }$   
16 Periodically resample collocation points and store checkpoint diagnostics.   
17 end for   
18 Return the trained approximation $Q _ { \theta } ( x )$

To optimize the FF–PINN implementation, we carried out the parameter sweeps summarized in Table 10, while fixing $p = 3$ . These tests varied the Fourier-feature dimension, Fourier scale, raw-coordinate inclusion, fixed versus trainable frequency matrices, optimizer schedule, collocation resolution, and refinement settings. To keep FF–PINN comparison consistent with the standard PINN baseline in section 3.1, while also using the best-performing FF–PINN settings from the sweeps, the working FF–PINN parameters were chosen with $N _ { f } = 2 ^ { 9 }$ total residual points, depth “ 4, and width “ 32.

The Dirichlet boundary condition formulation favors $N _ { \mathrm { F F } } = 3 2 $ and the SiLU activation. For this formulation, Adam–L–BFGS with $\sigma _ { \mathrm { F F } } ~ = ~ 0 . 2 5$ gave the best balance between accuracy and runtime. Although some more expensive refinements produced slightly smaller $L ^ { 2 }$ errors, the improvement was small compared with the additional CPU time. We therefore use the faster near-tie Adam–L–BFGS configuration as the current FF–PINN baseline. For the optimizer internals, the best Dirichlet results were obtained using an L–BFGS history size of 20 and a maximum of 30 inner iterations per outer step. The selected FF–PINN baseline configurations for our investigations are found in Table 11.

<table><tr><td>Parameter</td><td>Values tested / Description</td></tr><tr><td>Domain length L</td><td>30</td></tr><tr><td>Boundary condition</td><td>Dirichlet</td></tr><tr><td>Activation functions</td><td>tanh(x), SiLU(x), SiLUPLUS(x), e −x2 , and sech(x) , e−x4</td></tr><tr><td>Number of hidden layers</td><td>3, 4, or 5</td></tr><tr><td>Neurons per layer</td><td>32, 64, or 96</td></tr><tr><td>Optimizer</td><td>Adam, L-BFGS, GN/LM, Adam-L-BFGS, or Adam-GN/LM</td></tr><tr><td>Fourier feature count NFF</td><td>4, 8, 16, or 32</td></tr><tr><td>Fourier scale  $\sigma _ { \mathrm { F F } }$ </td><td>0.05, 0.1, 0.25, or 0.5</td></tr><tr><td>Raw coordinate input</td><td>Tested both with and without the raw coordinate x/L included</td></tr><tr><td>Fourier matrix B</td><td>Fixed or trainable</td></tr><tr><td>Number of residual points  $N _ { f }$ </td><td>28, 29, 210, or 211</td></tr><tr><td>Adam warm-up length</td><td>2,500, 5,000, 7,500, 10,000, 12,500, and 15,000 epochs</td></tr><tr><td>Adam learning rate lr</td><td>10−3 initial; StepLR decay</td></tr><tr><td>L-BFGS refinement steps</td><td>25, 50, 75, 100, 125, 150, or 200</td></tr><tr><td>L-BFGS history size</td><td>20, 50, or 100</td></tr><tr><td>L-BFGS max iterations per step</td><td>1, 5, 10, 20, or 30</td></tr><tr><td>L-BFGS learning rate</td><td>0.5 or 1.0</td></tr><tr><td>GN/LM refinement steps</td><td>25, 50, 75, 100, 125, 150, or 200</td></tr><tr><td>GN/LM initial damping</td><td> $5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , \mathrm { o r ~ } 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>GN/LM Maximum backtracking steps</td><td>2, 4, 6, 8, and 10</td></tr><tr><td>GN/LM Damping shrink factor</td><td>Values in [0.4, 1.1]</td></tr><tr><td>GN/LM Damping growth factor</td><td>5, 10, and 15</td></tr><tr><td>Selection criterion</td><td>Smallest  $L ^ { 2 }$  error, with runtime used to break near-ties</td></tr></table>

Table 10. FF–PINN architecture and optimization hyperparameters tested for solving (10) on r0, 30s.

<table><tr><td>Optimizer</td><td>Activation</td><td>Hidden Layers</td><td>Neurons per Layer</td><td> ${ \underline { { N _ { f } } } }$ </td><td>Adam Warm-up</td><td> $N _ { \mathrm { F F } }$ </td><td>σFF</td><td>B</td><td>Train Include L-BFGS Raw</td><td>Steps</td><td>History Size</td><td>Internal Iteration</td><td>L-BFGS LR</td></tr><tr><td>Adam-L-BFGS</td><td>SiLU(x)</td><td>4</td><td>32</td><td>29</td><td>2,500</td><td></td><td>32 0.25</td><td>yes</td><td>no</td><td>75</td><td>20</td><td>30</td><td>1.0</td></tr></table>

Table 11. Selected FF–PINN configurations for Dirichlet boundary conditions.

Additionally, we found that including the raw scaled coordinate x{L alongside the Fourier features did not give better accuracy than using only the sinusoidal embedding, so the following FF–PINN results do not include the raw input together with the trigonometric features.

FF–PINN with Dirichlet boundary conditions
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>1  T nrNr wrtn Dnicmet boumdary conartions</td></tr><tr><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1>LFFPINN(θ)</td><td rowspan=1 colspan=1>CPU time</td><td rowspan=1 colspan=1> $L ^ { \infty }$ error</td><td rowspan=1 colspan=1> $\overline { { L ^ { 2 } } }$ error</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1.756630e-07</td><td rowspan=1 colspan=1>219.333194</td><td rowspan=1 colspan=1>5.093381e-04</td><td rowspan=1 colspan=1>6.852868e-04</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1.004573e-07</td><td rowspan=1 colspan=1>200.046171</td><td rowspan=1 colspan=1>4.623447e-04</td><td rowspan=1 colspan=1>7.603941e-04</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>5.792016e-08</td><td rowspan=1 colspan=1>234.282703</td><td rowspan=1 colspan=1>3.989951e-04</td><td rowspan=1 colspan=1>7.267693e-04</td></tr></table>

Table 12. Comparison of FF–PINN performance with Dirichlet 16 boundary conditions (with parameters as in Table 11) for $Q ^ { \prime \prime } = Q - Q ^ { p }$

We find that the FF–PINN gives accurate approximations for all tested powers $p = 3 , 5 ,$ and 7. As shown in Table 12, the final losses remain on the order of $1 0 ^ { - 8 } \div 1 0 ^ { - 7 }$ , with $L ^ { \infty }$ errors below $5 . 1 \times 1 0 ^ { - 4 }$ and $L ^ { 2 }$ errors below $7 . 7 \times 1 0 ^ { - 4 }$ . The CPU times are comparable across the three cases, ranging from approximately 200 to 234 seconds. We obtain the smallest $L ^ { 2 }$ error for $p = 3$ , while the smallest $L ^ { \infty }$ error is obtained for $p = 7$ . The errors remain of the same order across all three tested powers.

![](images/3009760231968f52df7bf9b08058473a81e4fa7af89acd9c145be1aae20f1a2d.jpg)

![](images/ffcf9b3bbca7307c33435575cc596efb3ccfb72eb0b18379fc1a74bb9c649e72.jpg)  
Figure 6. Training loss (left) and ground state solution profile convergence (right) for FF–PINN (with parameters as seen in Table 11) with $Q ^ { \prime \prime } = Q - Q ^ { 5 }$ . The (black dashed) curve labeled Exact in the right panel is the explicit solution given by (14). The final learned profile corresponds to 2,500 Adam steps followed by 75 L–BFGS steps.

We mention that we use the term training loss in Figure 6 because ${ \mathcal { L } } _ { \mathrm { F F P I N N } } ( \theta )$ is the residual-based loss minimized by the FF–PINN. As described above, training objective is the more general term that we use when referring collectively to residual-based PINN and FF–PINN losses and the variational objectives used for the Deep Ritz methods.

In the left panel of Figure 6, we observe that the Adam warm-up stage steadily reduces the loss before the L–BFGS refinement stage further decreases it. In the right panel, we observe that the FF–PINN profiles rapidly converge to the exact ground state give by (14). By the final iterate, the numerical and exact curves are visibly indistinguishable on the plotted scale. The remaining discrepancy is consistent with the $p = 5$ errors reported in Table 12, which is orders $1 0 ^ { - 4 }$ . The representative convergence plot in Figure 6 illustrates this behavior for $p = 5 \colon$ we see the gradual decrease during the Adam stage followed by the final L–BFGS refinement in the left panel, while the final profile in the right panel aligns closely with the exact solution throughout the plotted domain.

3.2.1. Comparison with the “secondary” computational device for FF–PINN. The FF–PINN experiments in this section use the same baseline hyperparameters selected in Table 11, but are run on the secondary GPU-capable workstation. No additional tuning was performed for these runs. In particular, the Fourierfeature dimension, Fourier-feature scaling, raw-coordinate inclusion, activation function, network archi tecture, collocation size, boundary-condition weight, Adam warm-up, and refinement parameters were all kept fixed to the previously selected FF–PINN baseline. The purpose of this test was to check whether the selected FF–PINN baseline gives comparable accuracy when moved to the $\mathrm { G P U + C P U }$ workstation, rather than to search for a new optimal configuration on the secondary device.

Table 13 reports the FF–PINN results on the secondary GPU+CPU device. We observed that the method produces accurate ground-state approximations for all three power cases $p = 3 , 5 , 7 .$ , with the learned profiles remaining positive, centered at $x = 0$ , and visually consistent with the exact solution given by (14). The errors decrease as p increases, with the smallest $L ^ { \infty }$ and $L ^ { 2 }$ errors obtained for $p = 7$

From Table 13, we observe that the $L ^ { \infty }$ errors remain of order $1 0 ^ { - 4 }$ for all three powers. The $L ^ { 2 }$ errors are also of order $1 0 ^ { - 4 }$ for $p = 5$ and $p = 7$ , while the $p = 3$ value is of order $1 0 ^ { - 3 }$ . Thus, the errors decrease with $p ,$ and the fixed FF–PINN configuration continues to approximate the exact ground state accurately across the three tested powers. We also observe that GPU+CPU time remains comparable across the three cases, ranging from approximately 67 to 72 seconds.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>FF-PINN with Dirichlet boundary conditions</td></tr><tr><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1> ${ \mathcal { L } } _ { \mathrm { F F P I N N } } ( \theta )$ </td><td rowspan=1 colspan=1>GPU+CPU time</td><td rowspan=1 colspan=1> $L ^ { \infty }$ error</td><td rowspan=1 colspan=1> $L ^ { 2 }$ error</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>1.284711e-07</td><td rowspan=1 colspan=1>67.670952</td><td rowspan=1 colspan=1>7.727327e-04</td><td rowspan=1 colspan=1>1.107474e-03</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1.017865e-07</td><td rowspan=1 colspan=1>67.329122</td><td rowspan=1 colspan=1>4.419572e-04</td><td rowspan=1 colspan=1>7.254083e-04</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6.116914e-08</td><td rowspan=1 colspan=1>72.062521</td><td rowspan=1 colspan=1>2.975830e-04</td><td rowspan=1 colspan=1>5.681980e-04</td></tr></table>

Table 13. Secondary-device GPU+CPU comparison of FF–PINN performance with Dirichlet boundary conditions 16 (with parameters as in Table 11) for $Q ^ { \prime \prime } = Q - Q ^ { p }$

Compared with the corresponding primary-device results in Table 12, the secondary device runs have substantially have shorter the wall-clock times. In particular, the secondary-device times of approximately 67–72 seconds are considerably smaller than the approximately 200–234-second primary-device times. For $p = 3$ , the primary-device run gives smaller $L ^ { \infty }$ and $L ^ { 2 }$ errors, whereas for $p = 5$ and $p = 7$ , the secondarydevice runs give smaller errors for both $L ^ { \infty }$ and $L ^ { 2 }$ . Although the two workstations produce somewhat diferent numerical results, the errors remain of comparable magnitude across the two devices, indicating that the selected FF–PINN configuration gives similar accuracy on both systems while the secondarydevice runs require less wall-clock time. Unlike the higher-order optimization PINN secondary-device runs, the FF–PINN experiments do not show evidence of convergence to a non-ground-state branch; some numerical variation between the two workstation runs is expected because of diferent hardware, back-end execution, and other numerical details, but these diferences do not substantially change the accuracy of the FF–PINN solutions.

Overall, we find that the secondary-device FF–PINN comparison supports the stability of the selected FF–PINN baseline: when the hyperparameters are held fixed, the GPU+CPU implementation preserves the same accuracy behavior observed on the primary CPU device while achieving shorter run times.

## 4. Deep Ritz Variations

In this section we study the Deep Ritz formulations, which difer substantially from the residual-based PINN and FF–PINN methods studied above. In the PINN-type methods, the neural network is trained by minimizing a least-squares loss built directly from the pointwise ODE residual together with the relevant boundary residuals. By contrast, the Deep Ritz method is variational: the neural-network approximation $Q _ { \theta }$ of the profile Q is trained by minimizing a scalar objective derived from the variational structure of the ground state equation, rather than from the equation itself. This changes both the optimization landscape and the interpretation of the reported training objective.

The variational formulation cannot be minimized naively over an unconstrained neural-network class. This is because in the action-based formulation, the negative nonlinear term (see, for example, (40)) can drive the action away from the desired ground-state branch unless additional structure is imposed. Therefore, we use constrained Deep Ritz formulations, which are designed to select the positive, localized ground-state profile. The constraints that we use include hard boundary conditions, prescribed peak-amplitude information, positivity or branch selection, and, in the Nehari and Weinstein variants, additional integral constraints derived from the structure of the ground state equation.

A common feature of all Deep Ritz variants considered below is that the Adam training stage and the GN/LM refinement optimize diferent quantities. During the Adam stage, the method minimizes the scalar training objective associated with the selected formulation: the constrained action-based Deep Ritz loss, the Nehari-constrained objective, or the Weinstein quotient objective. These quantities are variational objectives rather than pointwise residual losses, thus, their final values are not expected to vanish even when the computed profile is accurate.

Here, GN/LM is used only as a final residual-based refinement stage. Accordingly, the GN/LM steps act on a separate, formulation-dependent least-squares residual vector. We define the corresponding refinement objective by

$$
\mathcal { T } _ { \mathrm { G N } } ( \theta ) = \frac { 1 } { 2 } \Vert F _ { \mathrm { G N } } ( \theta ) \Vert _ { l ^ { 2 } } ^ { 2 } ,\tag{39}
$$

where $F _ { \mathrm { G N } } ( \theta )$ is a finite-dimensional residual vector. Note that it has the same least-squares structure as the PINN residual vector (31), but its entries depend on the Deep Ritz formulation. For the constrained and Nehari-constrained Deep Ritz methods, it contains the residual components specified in Algorithm 3 and 4, including, where applicable, the ODE, boundary-condition, non-negativity, and normalized Pokhozhaev residuals. For the Weinstein formulation, $F _ { \mathrm { G N } } ( \boldsymbol { \theta } )$ instead consists of quadrature weighted values of the Euler-Lagrange residual of the normalized profile

$$
r _ { v , \theta } ( x ) = - v _ { \theta } ^ { \prime \prime } ( x ) + b v _ { \theta } ( x ) - R _ { b , L } ( \theta ) | v _ { \theta } ( x ) | ^ { p - 1 } v _ { \theta } ( x ) ,
$$

evaluated at the GN/LM residual points. Thus, the Weinstein refinement acts on the normalized profile v<sub>θ</sub> rather than directly on the ODE residual of the reconstructed ground-state profile $Q _ { \theta }$ . By (66) the two residuals difer only by the positive constant factor $( R _ { b , L } ( \theta ) / \gamma ) ^ { 1 / ( p - 1 ) }$ , so they have the same zero set, but the choice does afect the scaling of the Jacobian relative to a fixed damping parameter λ.

Throughout this section the Adam-stage quantity and the GN/LM stage quantity should be interpreted separately: the former measures the variational training objective of the selected Deep Ritz formulation, while the latter measures the residual-refinement objective. This distinction is especially important when interpreting the training-objective curves and the reported values of the Deep Ritz variational objectives and the separate GN/LM refinement objective $\mathcal { I } _ { \mathrm { G N } }$

We point out that the comparisons below concern complete, separately tuned training procedures (not a controlled comparison in which only the variational objective is changed). Every reported Deep Ritz configuration includes GN/LM refinement, and the methods difer in trial parameterization, architecture, penalties, and optimization budget. Thus, the final errors and runtimes cannot be attributed to the variational pretraining alone. Instead we wanted to show a more complete comparison of how well the methods could perform.

4.1. Constrained Deep Ritz Formulation. To find the (unique) positive ground state branch, we use an action-based Deep Ritz formulation. Instead of training directly on the pointwise ODE residual, we use the action associated with the ground state profile equation (11): for a profile Q on r0, Ls, the action is defined as

$$
S _ { b } ( Q ) = \int _ { 0 } ^ { L } \left( \frac { 1 } { 2 } | \partial _ { x } Q | ^ { 2 } + \frac { b } { 2 } Q ^ { 2 } - \frac { \gamma } { p + 1 } | Q | ^ { p + 1 } \right) d x .\tag{40}
$$

The Deep Ritz stage performs local action-based optimization over a neural-network trial, satisfying the prescribed boundary values, we supplement the action with Pokhozhaev identity penalties.

The boundary conditions $Q _ { \theta } ( 0 ) = Q _ { 0 }$ and $Q _ { \theta } ( L ) = 0$ are used to follow the positive ground state branch and are enforced exactly by the trial function below. Although these conditions alone do not make $\boldsymbol { S } _ { b }$ lower bounded, we show below that the prescribed height together with the Pokhozhaev penalties makes the desired ground state a strict local minimizer of the corresponding half-line objective.

For the Dirichlet formulation, we use the hard-constrained trial function

$$
Q _ { \theta } ( x ) = \underbrace { Q _ { 0 } \left( 1 - { \frac { x } { L } } \right) } _ { \mathrm { f i r s t ~ t e r m } } + \underbrace { { \frac { x } { L } } \left( 1 - { \frac { x } { L } } \right) q _ { \theta } \left( { \frac { x } { L } } \right) } _ { \mathrm { t r a i n a b l e ~ c o r r e c t i o n } } .\tag{41}
$$

This ansatz enforces the boundary conditions at 0 and at L exactly for all trainable parameters. Indeed, the first term satisfies the prescribed endpoint values $( Q _ { 0 }$ or 0), while the second term is a trainable

correction as we indicated in (41) that vanishes at both endpoints, thus, putting them together we get $Q _ { \theta } ( 0 ) = Q _ { 0 } + 0 = Q _ { 0 }$ and $Q _ { \theta } ( L ) = 0 + 0 = 0$ . Therefore, for the hard Dirichlet trial function in (41), the boundary residual is identically zero $( \mathrm { i . e . , }$ it leaves these boundary values unchanged).

One can observe that this trial function does not enforce positivity, monotonicity, or an upper bound $Q _ { \theta } \leqslant Q _ { 0 }$ , in particular, the network can take $q _ { \theta } \equiv t$ in (41), giving

$$
Q _ { t } ( x ) = Q _ { 0 } \left( 1 - \frac { x } { L } \right) + t \frac { x } { L } \left( 1 - \frac { x } { L } \right) , \quad \mathrm { a n d ~ t h u s , } \quad S _ { b } ( Q _ { t } ) \to - \infty \quad ( \mathrm { a s ~ } t \to \infty ) ,
$$

and furthermore, the normalized Pokhozhaev penalty in Algorithm 3 remains bounded along this family, and thus, for every fixed finite penalty weight, the total objective also tends to ´8 along this family.

To explain why this constrained formulation works, we first set the constrained Deep Ritz objective to be

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D R } } ( \theta ) = \mathcal { S } _ { b } ( Q _ { \theta } ) + \lambda _ { \mathrm { P o k h } } \mathcal { L } _ { \mathrm { P o k h } } ( \theta ) , } \end{array}\tag{42}
$$

where ${ \mathcal { L } } _ { \mathrm { P o k h } } ( \theta )$ penalizes violations of the Pokhozhaev identities $( 2 0 ) - ( 2 1 )$ , which are evaluated on the domain $[ 0 , L ]$ . (Note that a boundary penalty ${ \mathcal { L } } _ { \mathrm { B C } } ( \theta )$ would normally appear in (42) as well, however, for the trial function in (41) it vanishes identically, so we drop it from the objective (42).)

Next, we note that the ground-state branch is searched via the prescribed central value and the Pokhozhaev penalties. Indeed, the condition

$$
Q _ { \theta } ( 0 ) \equiv Q _ { 0 } = \left( \frac { b ( p + 1 ) } { 2 \gamma } \right) ^ { 1 / ( p - 1 ) }
$$

excludes the trivial solution and fixes the correct ground state height, while the trainable correction determines the profile shape. Moreover, the amplitude variation $Q \mapsto t Q$ is no longer admissible. The two Pokhozhaev penalties encourage the integral equalities satisfied by the ground state, including

$$
\| Q _ { \theta } ^ { \prime } \| _ { L ^ { 2 } ( 0 , L ) } ^ { 2 } \approx \frac { p - 1 } { p + 3 } b \| Q _ { \theta } \| _ { L ^ { 2 } ( 0 , L ) } ^ { 2 } .
$$

Thus, the boundary value provides the correct height, while the variational objective and the integral penalties guide the remaining shape approximation.

We now give a more precise local variational argument. For this calculation, consider the exact ground state $Q$ on $( 0 , \infty )$ , and interpret $\mathcal { S } _ { b } , \mathcal { P } _ { 1 } , \mathcal { P } _ { 2 }$ , and $\mathcal { L } _ { \mathrm { D R } }$ from (40), (20), (21) and (42), correspondingly, as their counterparts defined on the half line, i.e., with all integrals taken over $( 0 , \infty )$ . Then perturbations (preserving the prescribed height) have the form $Q + h$ , where $h \in H ^ { 1 } ( 0 , \infty )$ is real-valued and satisfies $h ( 0 ) = 0$ . Since $Q$ solves the profile equation (11), the first variation of the action vanishes for every such perturbation, namely,

$$
\displaystyle { S _ { b } ^ { \prime } ( Q ) [ h ] = \int _ { 0 } ^ { \infty } \left( Q ^ { \prime } h ^ { \prime } + b Q h - \gamma Q ^ { p } h \right) d x = 0 . }
$$

Thus, $Q$ is a critical point of the action among profiles with the prescribed boundary value at the origin. Diferentiating the profile equation gives

$$
\left( - \partial _ { x } ^ { 2 } + b - p \gamma Q ^ { p - 1 } \right) Q ^ { \prime } = 0 .\tag{43}
$$

Recall that for the branch that we are looking for, it holds $Q ^ { \prime } ( x ) < 0$ for $x > 0$ . Then integration by parts yields the following representation of the second variation

$$
S _ { b } ^ { \prime \prime } ( Q ) [ h , h ] = \int _ { 0 } ^ { \infty } \left( | h ^ { \prime } | ^ { 2 } + \left( b - p \gamma Q ^ { p - 1 } \right) h ^ { 2 } \right) d x = \int _ { 0 } ^ { \infty } ( Q ^ { \prime } ) ^ { 2 } \left| \left( \frac { h } { Q ^ { \prime } } \right) ^ { \prime } \right| ^ { 2 } d x \geqslant 0 .
$$

(To justify this, we first consider smooth perturbations supported away from the endpoints and then extend by density to $h \in H ^ { 1 } ( 0 , \infty )$ with $h ( 0 ) = 0 . )$ Note that the equality holds precisely when h is a multiple of $Q ^ { \prime }$ . Thus, prescribing the central value removes the negative quadratic directions of the action, leaving only this one-dimensional degeneracy $\left( \mathrm { i . e . , ~ s p a n } \{ Q ^ { \prime } \} \right)$

To control this remaining zero direction, we invoke the first Pokhozhaev penalty: using the profile equation, we obtain

$$
\mathcal { P } _ { 1 } ^ { \prime } ( Q ) [ h ] = 2 \int _ { 0 } ^ { \infty } \left( Q ^ { \prime } h ^ { \prime } + b Q h \right) d x - ( p + 1 ) \gamma \int _ { 0 } ^ { \infty } Q ^ { p } h d x = - ( p - 1 ) \gamma \int _ { 0 } ^ { \infty } Q ^ { p } h d x ,
$$

and hence,

$$
\mathcal { P } _ { 1 } ^ { \prime } ( Q ) [ Q ^ { \prime } ] \frac { \gamma ( p - 1 ) } { p + 1 } Q _ { 0 } ^ { p + 1 } > 0 .
$$

Since $\mathcal { P } _ { 1 } ( Q ) = \mathcal { P } _ { 2 } ( Q ) = 0$ , the second variation of the penalized objective is

$$
\mathcal { L } _ { \mathrm { D R } } ^ { \prime \prime } ( Q ) [ h , h ] = { \cal S } _ { b } ^ { \prime \prime } ( Q ) [ h , h ] + \frac { 2 \lambda _ { \mathrm { P o k h } } } { s ( Q ) ^ { 2 } } \left( | { \mathcal P } _ { 1 } ^ { \prime } ( Q ) [ h ] | ^ { 2 } + | { \mathcal P } _ { 2 } ^ { \prime } ( Q ) [ h ] | ^ { 2 } \right) ,\tag{44}
$$

where

$$
s ( Q ) = b \| Q \| _ { L ^ { 2 } ( 0 , \infty ) } ^ { 2 } + \| Q ^ { \prime } \| _ { L ^ { 2 } ( 0 , \infty ) } ^ { 2 } + \gamma \| Q \| _ { L ^ { p + 1 } ( 0 , \infty ) } ^ { p + 1 } + \varepsilon > 0
$$

is the analog of the normalization scale $s _ { \theta } = b M _ { \theta } + K _ { \theta } + \gamma N _ { \theta } + \varepsilon$ for the half line, which we use in Algorithm 3, evaluated at the exact ground state. (In the implementation, we use $\varepsilon = 1 0 ^ { - 1 2 }$ , and the scale is computed from the current neural approximation.)

The first term in (44) is nonnegative, and its only zero direction is detected by the first penalty. Consider the linearized operator $- \partial _ { x } ^ { 2 } + b - p \gamma Q ^ { p - 1 }$ on $L ^ { 2 } ( 0 , \infty )$ , with domain $\{ h \in H ^ { 2 } ( 0 , \infty ) : h ( 0 ) = 0 \}$ The boundary condition $h ( 0 ) = 0$ corresponds to perturbations that preserve the height $Q ( 0 ) = Q _ { 0 }$ . The identity (43) shows that this operator is nonnegative and has kernel span $\{ Q ^ { \prime } \}$ . Since $Q ^ { p - 1 } ( x )  0$ as $x \to \infty$ , its essential spectrum is $[ b , \infty )$ . Therefore, zero is a simple isolated eigenvalue, and the quadratic form is bounded below by a positive multiple of $\| h \| _ { L ^ { 2 } ( 0 , \infty ) } ^ { 2 }$ on the $L ^ { 2 }$ -orthogonal complement of $Q ^ { \prime }$ . Thus, for every fixed $\lambda _ { \mathrm { P o k h } } > 0$ , there exists $c _ { \lambda } > 0$ such that

$$
\begin{array} { r } { \mathcal L _ { \mathrm { D R } } ^ { \prime \prime } ( Q ) [ h , h ] \geqslant c _ { \lambda } \| h \| _ { H ^ { 1 } ( 0 , \infty ) } ^ { 2 } \quad \mathrm { w i t h } \quad h ( 0 ) = 0 . } \end{array}
$$

Then using the Taylor expansion, for suficiently small $\| h \| _ { H ^ { 1 } ( 0 , \infty ) }$ , we obtain

$$
{ \mathcal { L } } _ { \mathrm { D R } } ( Q + h ) - { \mathcal { L } } _ { \mathrm { D R } } ( Q ) \geqslant { \frac { c _ { \lambda } } { 4 } } \| h \| _ { H ^ { 1 } ( 0 , \infty ) } ^ { 2 } .
$$

In particular, $Q$ is an isolated strict local minimizer of the penalized half-line objective $\mathcal { L } _ { \mathrm { D R } }$ among profiles with the height $Q ( 0 ) = Q _ { 0 }$ . (In our implementation, the normalization denominator is recomputed from the current neural approximation at each evaluation but treated as constant when computing derivatives. Although this omits derivatives of the denominator, the resulting gradient has the same linearization at $Q$ as the gradient of the fully diferentiated objective, since both Pokhozhaev residuals vanish there.)

In summary, the given height $Q _ { 0 }$ together with the Pokhozhaev penalties is suficient to select the desired ground state locally, for every $p > 1$ (including the cubic, quintic, and septic cases that we consider here). For the finite-interval problem, functions satisfying $u ( L ) = 0$ can be extended by zero to the half-line, and a smooth cutof of $Q$ near L gives an admissible approximation, whose $H ^ { 1 }$ error is exponentially small as the length $L \to \infty$ . The local coercivity then gives a nearby local minimizer of the continuous truncated objective for suficiently large L. This gives an analytical justification for the local action-based training. In our computations, Adam pretraining produces a profile near the desired ground state, and GN/LM subsequently refines it by reducing the ODE residual, the normalized Pokhozhaev residuals, and a nonnegativity penalty while retaining the prescribed boundary values. The agreement with the ground state is verified by Figure 7 and the profile errors in Table 16.

This method can be referred to as a constrained Deep Ritz formulation with ground-state branch selection (rather than a standard residual-based PINN). Our implementation of this DR method, which we use in the numerical experiments, is summarized in Algorithm 3.

We mention that we use the whole line expressions $P _ { 1 , \theta }$ and $P _ { 2 , \theta }$ in Algorithm 3 as ground state regularizers, evaluated by quadrature on $[ 0 , L ]$ as ground state regularizers rather than exact finite interval identities. For the trial function (41), the omitted boundary terms are $Q _ { 0 } Q _ { \theta } ^ { \prime } ( 0 )$ and $L | Q _ { \theta } ^ { \prime } ( L ) | ^ { 2 }$ , neither of which is forced to vanish by the given boundary values. For comparison, the exact benchmark satisfies $Q _ { \mathrm { e x a c t } } ^ { \prime } ( 0 ) = 0$ and $Q _ { \mathrm { e x a c t } } ^ { \prime } ( x ) = - \sqrt { b } Q _ { \mathrm { e x a c t } } ( x )$ tanhpβxq. At $b = \gamma = 1$ $L = 3 0$ (and $p = 3 , 5 , 7 )$ , we have $Q _ { \mathrm { e x a c t } } ( L ) < 3 \times 1 0 ^ { - 1 3 }$ and $L | Q _ { \mathrm { e x a c t } } ^ { \prime } ( L ) | ^ { 2 } < 3 \times 1 0 ^ { - 2 4 }$ , therefore, these estimates motivate the use of the uncorrected expressions as regularizers for the intended whole line profiles. We note that, for example, our Nehari-constrained approach in Section $4 . 2 ,$ in contrast, retains the exact finite interval second Pokhozhaev identity correction.

Algorithm 3 Constrained Deep Ritz Method with Pokhozhaev Penalties   
1 Choose r0, Ls, quadrature points $\{ { x } _ { j } \} _ { j = 1 } ^ { N _ { q } } , b > 0 , \gamma > 0 , p > 1$ , and a Pokhozhaev penalty weight $\lambda _ { \mathrm { P o k h } } > 0 .$   
2 Compute the peak value $Q _ { 0 }$ from the condition   
$\frac { b } { 2 } Q _ { 0 } ^ { 2 } - \frac { \gamma } { p + 1 } Q _ { 0 } ^ { p + 1 } = 0 ,$   
whose positive root is $\begin{array} { r } { Q _ { 0 } = \left( \frac { b ( p + 1 ) } { 2 \gamma } \right) ^ { \frac { 1 } { p - 1 } } } \end{array}$ , in agreement with (15).   
3 Define a neural network q and the hard linear trial $Q _ { \theta }$ in (41), only $Q _ { \theta } ( 0 ) = Q _ { 0 }$ and $Q _ { \theta } ( L ) = 0$ are imposed exactly.   
4 Choose ${ \mathsf { O p t } } _ { \mathrm { D R } } \in \mathsf { \Gamma } .$ tAdam, L–BFGS, Adam Ñ L–BFGSu.   
5 for $n = 1 , 2 , \ldots , N _ { \mathrm { D R } }$ do   
6 Use automatic diferentiation and quadrature to compute   
$M _ { \theta } = \int _ { 0 } ^ { L } Q _ { \theta } ^ { 2 } d x , \quad K _ { \theta } = \int _ { 0 } ^ { L } | Q _ { \theta } ^ { \prime } | ^ { 2 } d x , \quad N _ { \theta } = \int _ { 0 } ^ { L } | Q _ { \theta } | ^ { p + 1 } d x .$   
7 Compute   
$S _ { b } ( Q _ { \theta } ) = \frac { 1 } { 2 } K _ { \theta } + \frac { b } { 2 } M _ { \theta } - \frac { \gamma } { p + 1 } N _ { \theta } .$   
8 Compute the whole-line Pokhozhaev expressions, truncated to r0, Ls, used as regularizers   
$P _ { 1 , \theta } = b M _ { \theta } + K _ { \theta } - \gamma N _ { \theta } , \qquad P _ { 2 , \theta } = b M _ { \theta } - K _ { \theta } - \frac { 2 \gamma } { p + 1 } N _ { \theta } .$   
9 Define   
$\mathcal { L } _ { \mathrm { P o k h } } = \left| \frac { P _ { 1 , \theta } } { s _ { \theta } } \right| ^ { 2 } + \left| \frac { P _ { 2 , \theta } } { s _ { \theta } } \right| ^ { 2 } , \qquad s _ { \theta } = b M _ { \theta } + K _ { \theta } + \gamma N _ { \theta } + \varepsilon .$   
where $s _ { \theta }$ nondimensionalizes the residuals by the magnitude of the individual bulk terms and $\varepsilon > 0$ is a numerical safeguard (its   
precise definition is formulation-dependent).   
10 Set   
$\mathcal { L } _ { \mathrm { D R } } =  { S _ { b } } ( Q _ { \theta } ) + \lambda _ { \mathrm { P o k h } }  { \mathcal { L } } _ { \mathrm { P o k h } }$   
The hard trial makes the boundary loss identically zero.   
11 Take a local optimizer step with $\mathsf { O p t } _ { \mathrm { D R } }$ on the fixed quadrature grid and store checkpoint diagnostics.   
12 end for   
13 if $\mathrm { G N / L M }$ refinement is selected then   
14 On at most $N _ { \mathrm { G N } }$ uniform points, form $F _ { \mathrm { G N } } ( \boldsymbol { \theta } )$ from the quadrature ODE residual, the two normalized Pokhozhaev residuals,   
and non-negativity residual.   
15 for $m = 1 , 2 , \ldots , N _ { \mathrm { G N / L M } }$ do   
16 Form $J _ { F } ( \theta ) = \partial F _ { \mathrm { G N } } / \partial \theta$ and solve the residual-space system $( J _ { F } J _ { F } ^ { T } + \mu I ) \alpha = F _ { \mathrm { G N } }$ , then set $\delta = - J _ { F } ^ { T } \alpha$ , with step clipping.   
17 Accept $\theta \gets \theta + \delta$ (after backtracking) when it decreases $\begin{array} { r } { \mathcal { T } _ { \mathrm { G N } } ( \theta ) = \frac { 1 } { 2 } \| F _ { \mathrm { G N } } ( \theta ) \| _ { l ^ { 2 } } ^ { 2 } } \end{array}$ , shrink $\mu$ on acceptance and grow it   
otherwise.   
18 end for   
19 end if   
20 Return the trained approximation $Q _ { \theta } .$

The terms “soft”, “hard”, and “decaying” in the Trial function line in Table 14 describe how the boundary conditions are imposed. In the soft formulation, the network output is unconstrained and the boundary conditions are enforced through penalty terms. In the hard formulation, the trial function is constructed so that the boundary conditions are satisfied exactly for every choice of the trainable parameters, as in (41). In the decay formulation, the trial function also satisfies the boundary conditions exactly, but uses the stronger built-in decay structure in (38) instead of the linear boundary interpolator. We tested all three formulations; however, the hard formulation with the linear constraint (41) produced the smallest errors, therefore, in the actual algorithm (see Algorithm 3) we drop the boundary condition term $\lambda _ { \mathrm { B C } } \mathcal { L } _ { \mathrm { B C } } ( \theta )$ from the loss $\mathcal { L } _ { \mathrm { D R } } ( \theta )$

As discussed above, the Adam stage minimizes the action-based Deep Ritz objective, while the optional GN/LM stage is used only as a residual-based refinement of the pretrained profile. Therefore, the GN/LM parameters in Table 14 should be interpreted as refinement parameters for the least-squares residua system, not as parameters of the scalar action minimization itself.

<table><tr><td>Parameter</td><td>Values tested / Description</td></tr><tr><td>Domain length L</td><td>30</td></tr><tr><td>Boundary condition</td><td>Dirichlet</td></tr><tr><td>Trial function</td><td>Soft penalty trial, hard or decaying Dirichlet ansatz</td></tr><tr><td>Optimizer</td><td>Adam, L-BFGS, or Adam-L-BFGS</td></tr><tr><td>Refinement strategy</td><td>GN/LM or Adam-GN/LM</td></tr><tr><td>Activation functions</td><td>and sech(x) , e−x4</td></tr><tr><td>Number of hidden layers</td><td>3, 4, or 5</td></tr><tr><td>Neurons per layer</td><td>32, 64, or 96</td></tr><tr><td>Number of quadrature points  $N _ { q }$ </td><td> $\phantom { - } 2 ^ { 8 } , 2 ^ { 9 } , 2 ^ { 1 0 } , 2 ^ { 1 1 } , \phantom { - } \mathrm { o r } 2 ^ { 1 2 }$ </td></tr><tr><td>Adam warm-up length</td><td>2,500, 5,000, 7,500, 10,000, 12,500, and 15,000 epochs</td></tr><tr><td>Adam learning rate lr</td><td> $1 0 ^ { - 3 }$  initial; StepLR decay</td></tr><tr><td>Pokhozhaev penalty weight λPokh</td><td>0.1, 1, 5, 10, or 25</td></tr><tr><td>L-BFGS refinement steps</td><td>25, 50, 75, 100, 125, 150, or 200</td></tr><tr><td>L-BFGS history size</td><td>20, 50, or 100</td></tr><tr><td>L-BFGS max iterations per step</td><td>1, 5, 10, 20, or 30</td></tr><tr><td>L-BFGS learning rate</td><td>0.5 or 1.0</td></tr><tr><td>GN/LM refinement steps</td><td>25, 50, 75, 100, 125, 150, or 200</td></tr><tr><td>GN/LM initial damping</td><td> $5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , \mathrm { o r ~ 2 } \times 1 0 ^ { - 3 }$ </td></tr><tr><td>GN/LM Maximum backtracking steps</td><td> $2 , 4 , 6 , 8 ,$  and 10</td></tr><tr><td>GN/LM Damping shrink factor</td><td>Values in [0.4, 1.1]</td></tr><tr><td>GN/LM Damping growth factor</td><td>5, 10, and 15</td></tr><tr><td>Selection criterion</td><td>Smallest  $L ^ { 2 }$  error, with runtime used to break near-ties</td></tr></table>

Table 14. Deep Ritz architecture, constraint, and optimization hyperparameters tested for solving (10) on r0, 30s.

The hyperparameter sweep in Table 14 was used to select the final Deep Ritz configurations. The collocation sweeps gave similar errors for $2 ^ { 8 } , 2 ^ { 9 } , 2 ^ { 1 0 } , 2 ^ { 1 1 }$ , and $2 ^ { 1 2 }$ action quadrature points, so $N _ { q } = 2 ^ { 9 }$ was retained. The GN/LM residual refinement uses an initial damping of $1 0 ^ { - 3 }$ , at most 8 backtracking steps, and a fixed maximum of 128 GN residual points to control memory usage. Increasing the GN/LM residual or quotient grid did not materially change the final reported $L ^ { \infty }$ or $L ^ { 2 }$ errors, but increasing either grid increased memory usage and runtime. Therefore, we retained the reduced 128-point GN/LM grid as a computationally eficient refinement grid, while verifying the final solution on an independent uniform evolution grid of 4096 points on $[ 0 , L ]$ , with the reported $L ^ { 2 }$ error evaluated by composite trapezoid quadrature. All reported Deep Ritz results therefore use the selected configurations in Table 15.

<table><tr><td colspan="10">Hidden Neurons Constraint Optimization Refinement Activation Layers per Layer Nq Warm-up</td><td colspan="7">Damping Damping</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>2,500</td><td>Steps 200</td><td>Backtracks</td><td>Shrink</td><td>Growth</td><td>Damping</td><td>Points</td><td>Residual Quotient Points</td><td>λPokh</td></tr><tr><td>Hard</td><td>Adam</td><td>GN/LM</td><td>SiLU(x)</td><td>5</td><td>32</td><td>29</td><td></td><td></td><td>8</td><td>0.7</td><td>10</td><td>10-3</td><td>128</td><td>128</td><td>25</td></tr></table>

Table 15. Selected Deep Ritz configurations for Dirichlet boundary conditions (16).

Table 16 reports the performance of the constrained Deep Ritz method for p “ 3, 5, and 7. The reported ${ \mathcal { L } } _ { \mathrm { D R } } ( \theta )$ values correspond to the final action-based Deep Ritz objective and are not expected to converge to zero. The residual quality after refinement is instead measured by ${ \mathcal { I } } _ { \mathrm { G N } } ( \theta )$ , which is the least-squares objective used in the GN/LM refinement stage.

We find that the constrained Deep Ritz method gives very accurate results for the cubic and quintic cases. For $p = 3$ , the method produces ${ \mathcal { I } } _ { \mathrm { G N } } ( \theta )$ of order $1 0 ^ { - 1 1 }$ , with $L ^ { \infty }$ and $L ^ { 2 }$ errors both of order $1 0 ^ { - 6 }$ For $p \ = \ 5$ the residual-refinement objective is reduced further to order $1 0 ^ { - 1 4 }$ , while the $L ^ { \infty }$ and $L ^ { 2 }$ errors are of $1 0 ^ { - 5 }$ and $1 0 ^ { - 6 } .$ respectively. The septic case $p = 7$ is slightly more challenging: although $\mathcal { I } _ { \mathrm { G N } }$ reaches order $1 0 ^ { - 1 6 }$ , the final errors increase to order $1 0 ^ { - 3 }$ . We interpret this as indicating that the sharper higher-power profile is harder to approximate accurately, even when the residual-based refinement objective is very small. In particular, these results show that a smaller value of ${ \mathcal { I } } _ { \mathrm { G N } } ( \theta )$ does not necessarily imply a smaller profile error across diferent values of $p .$

<table><tr><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1> ${ \mathcal { L } } _ { \mathrm { D R } } ( \theta )$ </td><td rowspan=1 colspan=1> ${ \mathcal { I } } _ { \mathrm { G N } } ( \theta )$ </td><td rowspan=1 colspan=1>CPU time</td><td rowspan=1 colspan=1> $\overline { { L ^ { \infty } } }$ error</td><td rowspan=1 colspan=1> $L ^ { 2 }$ error</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>6.666667e-01</td><td rowspan=1 colspan=1>9.620502e-11</td><td rowspan=1 colspan=1>264.092212</td><td rowspan=1 colspan=1>4.302695e-06</td><td rowspan=1 colspan=1>9.264227e-06</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6.801746e-01</td><td rowspan=1 colspan=1>2.375216e-14</td><td rowspan=1 colspan=1>252.688689</td><td rowspan=1 colspan=1>1.224911e-05</td><td rowspan=1 colspan=1>6.049316e-06</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6.679167e-01</td><td rowspan=1 colspan=1>4.063798e-16</td><td rowspan=1 colspan=1>246.570857</td><td rowspan=1 colspan=1>3.198027e-03</td><td rowspan=1 colspan=1>2.417198e-03</td></tr></table>

Table 16. Performance of the constrained Deep Ritz method with Dirichlet boundary conditions (16) for $Q ^ { \prime \prime } = Q - Q ^ { p }$

The CPU runtimes are relatively consistent across the three powers, ranging from around 246–264 seconds. Thus, we find that the main cost is determined primarily by the fixed network architecture, quadrature size, and GN/LM refinement settings, rather than by the particular value of $p .$

![](images/58832b6213fcc2de676d094d37a7af0c655a031e0e3e393d0c71afdfa676d416.jpg)

![](images/ae835157daeec44a48ab1dca498e188897b725ac25bf5c8d78344d8a821198ad.jpg)  
Figure 7. Training objective (left) and ground state solution profile convergence (right) for constrained Deep Ritz method (with parameters as seen in Table 15) for $Q ^ { \prime \prime } = Q - Q ^ { 5 }$ The (black dashed) curve labeled Exact in the right panel is the explicit solution given by (14). The final learned profile corresponds to 2,500 Adam steps followed by 200 GN/LM steps.

Figure 7 illustrates the two-stage behavior of the method. In the left panel, we observed that, during the Adam warm-up, the action-based Deep Ritz objective decreases and stabilizes near its final variational value. Once GN/LM refinement begins, the residual-based objective decreases by several additional orders of magnitude. In the right panel, we compare the learned profiles with the exact ground state given by (14). We observe that the Adam-trained approximation already visually captures the main ground-state shape, while the GN/LM stage visually improves the agreement with the exact solution. The final profile after 2,500 Adam steps and 200 GN/LM steps aligns closely with the exact curve throughout the plotted domain. The remaining discrepancy is consistent with the $p = 5$ errors in Table 16: the final $L ^ { \infty }$ error is $1 . 2 2 4 9 1 1 \times 1 0 ^ { - 5 }$ , while the $L ^ { 2 }$ error is $6 . 0 4 9 3 1 6 \times 1 0 ^ { - 6 }$

4.1.1. Comparison with the “secondary” computational device for constrained Deep Ritz. Table 17 reports the same constrained Deep Ritz configuration on the secondary $\mathrm { G P U + C P U }$ workstation. No additional hyperparameter tuning was performed on this device. The purpose of this test is to determine whether the selected constrained Deep Ritz baseline transfers reliably to a diferent computational environment and to quantify the corresponding runtime change. We observe that the secondary device substantially reduces the elapsed runtime for all three powers. The resulting primary-to-secondary runtime ratios correspond to speed-ups of approximately 6.4, 6.5, and 5.8 for $p = 3 , 5$ , and 7, respectively.
<table><tr><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1> ${ \mathcal { L } } _ { \mathrm { D R } } ( \theta )$ </td><td rowspan=1 colspan=1> ${ \mathcal { I } } _ { \mathrm { G N } } ( \theta )$ </td><td rowspan=1 colspan=1>GPU+CPU time</td><td rowspan=1 colspan=1> $L ^ { \infty }$ error</td><td rowspan=1 colspan=1> $L ^ { 2 }$ error</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>6.666667e-01</td><td rowspan=1 colspan=1>7.300457e-11</td><td rowspan=1 colspan=1>41.048747</td><td rowspan=1 colspan=1>3.697873e-06</td><td rowspan=1 colspan=1>7.913867e-06</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6.801749e-01</td><td rowspan=1 colspan=1>1.826449e-12</td><td rowspan=1 colspan=1>38.984522</td><td rowspan=1 colspan=1>2.828127e-05</td><td rowspan=1 colspan=1>1.107741e-05</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6.677846e-01</td><td rowspan=1 colspan=1>1.326228e-09</td><td rowspan=1 colspan=1>42.222312</td><td rowspan=1 colspan=1>1.397995e-03</td><td rowspan=1 colspan=1>1.007490e-03</td></tr></table>

Table 17. Secondary-device $\mathrm { G P U + C P U }$ performance of the constrained Deep Ritz method with Dirichlet boundary conditions (16) for $Q ^ { \prime \prime } = Q - Q ^ { p }$

The final errors on the secondary device remain comparable to those obtained on the primary device. For $p = 3$ , both the $L ^ { \infty }$ and $L ^ { 2 }$ errors improve slightly and remain of order $1 0 ^ { - 6 }$ , while for $p = 5$ , the secondary run give slightly larger errors: $L ^ { \infty }$ error remains of order $1 0 ^ { - 5 }$ , while the $L ^ { 2 }$ error increases from order $1 0 ^ { - 6 }$ to order $1 0 ^ { - 5 }$ . For $p = 7$ , both errors decrease relative to the primary-device values but remain of order $1 0 ^ { - 3 }$ . Thus, we find that the secondary-device runs preserve the same accuracy patterns as the primary device runs: strong accuracy for $p = 3$ and $p = 5$ , with a more challenging approximation for the sharper $p = 7$ profile.

We also observe that the relationship between the residual-refinement objective and the profile error is not monotone across the two devices. For example, for $p = 7$ , the secondary-device value of ${ \mathcal { I } } _ { \mathrm { G N } } ( \theta )$ is of order $1 0 ^ { - 9 }$ rather than $1 0 ^ { - 1 6 }$ , yet the secondary-device $L ^ { \infty }$ and $L ^ { 2 }$ errors are smaller. This again indicates that the magnitude of the sampled residual objective alone does not completely determine the error relative to the exact ground state (14).

Overall, we find that the constrained Deep Ritz method transfers well to the secondary GPU+CPU workstation. The main efect of the secondary device is a large runtime reduction, while the final errors remain comparable to the primary-device results. On the secondary device, we also observe that the constrained Deep Ritz method improves substantially over the corresponding FF–PINN results for $p = 3$ and $p = 5$ , while the comparison is less decisive for the sharper $p = 7$ case.

4.2. Nehari-Constrained Deep Ritz Formulation. For the ground-state solution which we consider on r0, Ls, of equation (11), we consider the associated action

$$
S _ { b } ( Q ) = \int _ { 0 } ^ { L } \left[ { \frac { 1 } { 2 } } | Q ^ { \prime } ( x ) | ^ { 2 } + { \frac { b } { 2 } } Q ( x ) ^ { 2 } - { \frac { \gamma } { p + 1 } } Q ( x ) ^ { p + 1 } \right] d x .\tag{45}
$$

The ground state profile equation (11) is the Euler–Lagrange equation for this action, $S _ { b } ^ { \prime } ( Q ) = 0$ . However, the focusing $( \gamma > 0 )$ action is not bounded from below on the full trial space. To show that, consider any nonzero u to get

$$
S _ { b } ( t u ) = \frac { t ^ { 2 } } { 2 } \int _ { 0 } ^ { L } \bigl ( | u ^ { \prime } | ^ { 2 } + b u ^ { 2 } \bigr ) d x - \gamma \frac { t ^ { p + 1 } } { p + 1 } \int _ { 0 } ^ { L } | u | ^ { p + 1 } d x \to - \infty \quad \mathrm { a s ~ } t \to \infty .
$$

Thus, the ground state cannot be described as an unconstrained minimizer of $\boldsymbol { S } _ { b }$ . In the previous section, in the constrained DR algorithm, we considered a specific branch, where the initial condition $Q _ { 0 } = A$ helped us locate the correct branch, which was possible since we knew the explicit solutions, and hence, the appropriate branch of solutions (although we also showed why fixing the $Q _ { 0 }$ gives the appropriate branch). However, if we did not have explicit solutions, and thus, the peak height explicitly, then we would need to add extra constraints or other conditions.

A possible constraint is, for example, to use one of the Pokhozhaev identities. Taking the first identity (20), sometimes also called the Nehari identity (and to distinguish it from the second Pokhozhaev identity, we denote it here via $\mathcal { N } _ { b } )$ , we define it on a finite interval r0, Ls

$$
\mathcal { N } _ { b } ( u ) : = \int _ { 0 } ^ { L } \left( | u ^ { \prime } | ^ { 2 } + b u ^ { 2 } - \gamma | u | ^ { p + 1 } \right) d x = 0 .\tag{46}
$$

The corresponding Nehari manifold is

$$
\mathcal { M } _ { L } = \{ u \in H ^ { 1 } ( 0 , L ) \backslash \{ 0 \} : \mathcal { N } _ { b } ( u ) = 0 \mathrm { ~ a n d ~ } u ( L ) = 0 \} .
$$

Then the nontrivial ground state is characterized variationally by minimizing $\boldsymbol { S } _ { b }$ over nonzero profiles on the Nehari manifold $\mathcal { M } _ { L }$ . With this constraint, we have

$$
S _ { b } ( u ) = \frac { p - 1 } { 2 ( p + 1 ) } \int _ { 0 } ^ { L } \left( | u ^ { \prime } | ^ { 2 } + b u ^ { 2 } \right) d x > 0 .\tag{47}
$$

Thus, the value of the action at the trained ground state is not expected to converge to zero.

To avoid prescribing the unknown amplitude Qp0q, we first construct a positive shape $\widetilde { Q } _ { \theta }$ with a unit height. For that we set

$$
z = \left( \frac { x } { L } \right) ^ { 2 } , \qquad \rho ( s ) = \log ( 1 + e ^ { s } ) ,
$$

and define

$$
\widetilde Q _ { \theta } ( x ) = ( 1 - z ) \frac { \rho \big ( q _ { \theta } ( z ) \big ) } { \rho \big ( q _ { \theta } ( 0 ) \big ) } .\tag{48}
$$

Then we get

$$
\widetilde Q _ { \theta } ( 0 ) = 1 , \qquad \widetilde Q _ { \theta } ^ { \prime } ( 0 ) = 0 , \qquad \widetilde Q _ { \theta } ( L ) = 0 , \qquad \widetilde Q _ { \theta } ( x ) > 0 \quad ( 0 \leqslant x < L ) .
$$

We note that the symmetry condition at the origin and the truncated Dirichlet condition at $x = L$ are now imposed exactly, while the amplitude remains to be determined by the variational problem.

Define

$$
\widetilde { M } _ { \theta } = \int _ { 0 } ^ { L } \widetilde { Q } _ { \theta } ^ { 2 } d x , \widetilde { K } _ { \theta } = \int _ { 0 } ^ { L } | \widetilde { Q } _ { \theta } ^ { \prime } | ^ { 2 } d x , \widetilde { N } _ { \theta } = \int _ { 0 } ^ { L } \widetilde { Q } _ { \theta } ^ { p + 1 } d x .\tag{49}
$$

Then the Nehari projection is obtained by setting

$$
A _ { \theta } = \left( \frac { \widetilde { K } _ { \theta } + b \widetilde { M } _ { \theta } } { \gamma \widetilde { N } _ { \theta } } \right) ^ { 1 / ( p - 1 ) } , \qquad Q _ { \theta } = A _ { \theta } \widetilde { Q } _ { \theta } .\tag{50}
$$

From scaling it follows that

$$
K _ { \theta } = A _ { \theta } ^ { 2 } \widetilde { K } _ { \theta } , \qquad M _ { \theta } = A _ { \theta } ^ { 2 } \widetilde { M } _ { \theta } , \qquad N _ { \theta } = A _ { \theta } ^ { p + 1 } \widetilde { N } _ { \theta } ,
$$

and thus, (50) gives

$$
K _ { \theta } + b M _ { \theta } - \gamma N _ { \theta } = 0 .
$$

Therefore, the first Pokhozhaev identity is enforced by the above Nehari projection, and we do not need to include it as a penalty in constrained Deep Ritz approach. The projected action can be evaluated as

$$
S _ { b } ( Q _ { \theta } ) = \frac { p - 1 } { 2 ( p + 1 ) } A _ { \theta } ^ { 2 } \bigl ( \widetilde { K } _ { \theta } + b \widetilde { M } _ { \theta } \bigr ) .\tag{51}
$$

For the truncated Dirichlet problem, the second Pokhozhaev identity contains a boundary correction as we discussed in §2.4, thus, it becomes $\begin{array} { r } { P _ { 2 , L } ( Q ) : = b M ( Q ) - K ( Q ) - \frac { 2 \gamma } { p + 1 } N ( Q ) + L | Q ^ { \prime } ( L ) | ^ { 2 } = 0 } \end{array}$ . Note

that the last term vanishes completely only in the infinite-domain. If we omit it, it may introduce a small truncation error. We therefore use

$$
\mathcal { L } _ { \mathrm { P o k h } } ( \theta ) = \left| \frac { P _ { 2 , L } ( Q _ { \theta } ) } { s _ { \theta } } \right| ^ { 2 } , \quad \mathrm { w h e r e } \quad s _ { \theta } = b M _ { \theta } + K _ { \theta } + \frac { 2 \gamma } { p + 1 } N _ { \theta } + L | Q _ { \theta } ^ { \prime } ( L ) | ^ { 2 } + \varepsilon .\tag{52}
$$

The Nehari-constrained Deep Ritz loss is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { N D R } } ( \theta ) = \mathcal { S } _ { b } ( Q _ { \theta } ) + \lambda _ { \mathrm { P o k h } } \mathcal { L } _ { \mathrm { P o k h } } ( \theta ) . } \end{array}\tag{53}
$$

We note that no boundary penalty is required for the hard-constrained ansatz (48).

```latex
Algorithm 4 Nehari-Constrained Deep Ritz Method for $Q ^ { \prime \prime } = b Q - \gamma Q ^ { p }$
1 Choose $b > 0 , \gamma > 0 , p > 1 ,$ the domain $[ 0 , L ] ,$ , quadrature points $\{ x _ { j } \} _ { j = 1 } ^ { N _ { q } } ,$ and $\lambda _ { \mathrm { P o k h } } \geqslant 0 .$
2 Define a neural network $q _ { \theta } ( z )$ for $z \in [ 0 , 1 ]$ and set $\rho ( s ) = \log ( 1 + e ^ { s } )$
3 Define the positive hard-constrained shape
$z = \left( \frac { x } { L } \right) ^ { 2 } , \qquad \tilde { Q } _ { \theta } ( x ) = ( 1 - z ) { \frac { \rho ( q _ { \theta } ( z ) ) } { \rho ( q _ { \theta } ( 0 ) ) } } .$
4 Choose the variational optimizer
$\begin{array} { r } { \mathbf { O p t } _ { \mathrm { D R } } \in \{ \mathrm { A d a m } , \mathrm { ~ L } \mathrm { - B F G S } , \mathrm { ~ A d a m }  \mathrm { L } \mathrm { - B F G S } \} . } \end{array}$
5 for $n = 1 , 2 , \ldots , N _ { \mathrm { e p o c h s } }$ do
6 Use automatic diferentiation and quadrature to compute
$\widetilde { M } _ { \theta } = \int _ { 0 } ^ { L } \widetilde { Q } _ { \theta } ^ { 2 } d x , \quad \widetilde { K } _ { \theta } = \int _ { 0 } ^ { L } | \widetilde { Q } _ { \theta } ^ { \prime } | ^ { 2 } d x , \quad \widetilde { N } _ { \theta } = \int _ { 0 } ^ { L } \widetilde { Q } _ { \theta } ^ { p + 1 } d x .$
7 Compute the diferentiable Nehari amplitude
$A _ { \theta } = \left( \frac { \widetilde { K } _ { \theta } + b \widetilde { M } _ { \theta } } { \gamma \widetilde { N } _ { \theta } } \right) ^ { 1 / \left( p - 1 \right) }$
and set $Q _ { \theta } = A _ { \theta } \tilde { Q } _ { \theta } .$
8 Keep $A _ { \theta }$ in the automatic-diferentiation graph; do not detach it from the trainable parameters.
9 Compute
$M _ { \theta } = { \cal A } _ { \theta } ^ { 2 } \widetilde { { \cal M } } _ { \theta } , \qquad K _ { \theta } = { \cal A } _ { \theta } ^ { 2 } \widetilde { { \cal K } } _ { \theta } , \qquad { \cal N } _ { \theta } = { \cal A } _ { \theta } ^ { p + 1 } \widetilde { { \cal N } } _ { \theta } .$
10 Compute the projected action
$S _ { b } ( Q _ { \theta } ) = \frac { p - 1 } { 2 ( p + 1 ) } A _ { \theta } ^ { 2 } \bigl ( \widetilde { K } _ { \theta } + b \widetilde { M } _ { \theta } \bigr ) .$
11 Compute the finite-interval Pokhozhaev residual
$P _ { 2 , L , \theta } = b M _ { \theta } - K _ { \theta } - \frac { 2 \gamma } { p + 1 } N _ { \theta } + L \left( Q _ { \theta } ^ { \prime } ( L ) \right) ^ { 2 }$
and the normalized penalty
$\mathcal { L } _ { \mathrm { P o k h } } ( \theta ) = \left| \frac { P _ { 2 , L , \theta } } { s _ { \theta , L } } \right| ^ { 2 } ,$
where
$s _ { \theta , L } = b M _ { \theta } + K _ { \theta } + \frac { 2 \gamma } { p + 1 } N _ { \theta } + L \left( Q _ { \theta } ^ { \prime } ( L ) \right) ^ { 2 } + \varepsilon ,$
and $\varepsilon > 0$ is a small numerical safeguard (the precise definition of the normalization scale $s _ { \boldsymbol { \theta } , L }$ is formulation-dependent).
12 Form
$\begin{array} { r } { \mathcal { L } _ { \mathrm { N D R } } ( \theta ) = \mathcal { S } _ { b } ( Q _ { \theta } ) + \lambda _ { \mathrm { P o k h } } \mathcal { L } _ { \mathrm { P o k h } } ( \theta ) . } \end{array}$
13 Update θ using ${ \mathrm { O p t } } _ { \mathrm { D R } } .$
14 Using the fixed quadrature grid, store $S _ { b } ( Q _ { \theta } ) , \mathcal { L } _ { \mathrm { P o k h } } ( \theta )$ , and $| K _ { \theta } + b M _ { \theta } - \gamma N _ { \theta } | .$
15 end for
16 Evaluate the pointwise ODE diagnostic
$\mathcal { L } _ { \mathrm { O D E } } ( \theta ) = \frac { 1 } { N _ { r } } \sum _ { j = 1 } ^ { N _ { r } } \left| Q _ { \theta } ^ { \prime \prime } ( x _ { j } ) - b Q _ { \theta } ( x _ { j } ) + \gamma Q _ { \theta } ( x _ { j } ) ^ { p } \right| ^ { 2 } .$
17 Optionally apply a $\mathrm { G N / L M }$ residual refinement while retaining the hard boundary conditions and the Nehari projection.
18 Return $Q _ { \theta } ,$ together with $S _ { b } ( Q _ { \theta } ) , \mathcal { L } _ { \mathrm { O D E } } ( \theta ) , \mathcal { L } _ { \mathrm { P o k h } } ( \theta )$ , and the Nehari residual.
```

After the variational training stage, we evaluate the pointwise equation residual

$$
\mathcal { R } _ { \theta } ( x ) = Q _ { \theta } ^ { \prime \prime } ( x ) - b Q _ { \theta } ( x ) + \gamma Q _ { \theta } ( x ) ^ { p } , \qquad \mathcal { L } _ { \mathrm { O D E } } ( \theta ) = \frac { 1 } { N _ { r } } \sum _ { j = 1 } ^ { N _ { r } } | \mathcal { R } _ { \theta } ( x _ { j } ) | ^ { 2 } .\tag{54}
$$

In this formulation, the network first represents a positive shape with a unit height, satisfying the hard boundary and symmetry constraints, and the physical amplitude is then determined through the Nehari projection. Thus, unlike the constrained Deep Ritz formulation above, the peak value $Q ( 0 )$ is not prescribed from the exact solution. Instead, the amplitude is selected variationally through (50), which enforces the Nehari identity up to quadrature and floating-point error.

Remark 1. The Nehari-projected action is directly related to the Weinstein quotient. Eliminating $A _ { \theta }$ from (51) gives

$$
S _ { b } ( Q _ { \theta } ) = \frac { p - 1 } { 2 ( p + 1 ) } \gamma ^ { - \frac { 2 } { p - 1 } } \left[ \frac { \widetilde { K } _ { \theta } + b \widetilde { M } _ { \theta } } { ( \widetilde { N } _ { \theta } ) ^ { \frac { 2 } { p + 1 } } } \right] ^ { \frac { p + 1 } { p - 1 } } .\tag{55}
$$

Hence, the Nehari-constrained DR method and the Weinstein DR method produce equivalent minimizers, in the sense that when $\lambda _ { \mathrm { P o k h } } = 0$ and the same raw shape and quadrature are used, the Nehari-projected action is a strictly increasing function of the Weinstein quotient and they have the same minimizing shapes. (We mention that our reported NDR and WDR procedures nevertheless difer through the additional second-identity penalty in NDR, the residuals used in GN/LM refinement, and their separately selected parameterizations and hyperparameters.)

<table><tr><td>Parameter</td><td>Values tested / Description</td></tr><tr><td>Domain length L</td><td>30</td></tr><tr><td>Boundary condition</td><td>mixed</td></tr><tr><td>Trial function</td><td>Soft penalty trial, hard or decaying Dirichlet ansatz</td></tr><tr><td>Optimizer</td><td>Adam, L-BFGS, or Adam-L-BFGS</td></tr><tr><td>Refinement strategy</td><td>GN/LM or Adam-GN/LM</td></tr><tr><td>Activation functions</td><td> $e ^ { - x ^ { 2 } } , e ^ { - x ^ { 4 } }$  tanh(x), SiLU(x), SiLUPLUS(x), , and sech(x)</td></tr><tr><td>Number of hidden layers</td><td>3, 4, or 5</td></tr><tr><td>Neurons per layer</td><td>32, 64, or 96</td></tr><tr><td>Number of quadrature points  $N _ { q }$ </td><td> $\phantom { - } 2 ^ { 8 } , 2 ^ { 9 } , 2 ^ { 1 0 } , 2 ^ { 1 1 } , \phantom { - } \mathrm { o r } 2 ^ { 1 2 }$ </td></tr><tr><td>Adam warm-up length</td><td>2,500, 5,000, 7,500, 10,000, 12,500, or 15,000 steps</td></tr><tr><td>Adam learning rate lr</td><td> $1 0 ^ { - 3 }$  initial; StepLR decay</td></tr><tr><td>Pokhozhaev penalty weight λPokh</td><td>0.1, 1, 5, 10, or 25</td></tr><tr><td>L-BFGS refinement steps</td><td>25, 50, 75, 100, 125, 150, or 200</td></tr><tr><td>L-BFGS history size</td><td>20, 50, or 100</td></tr><tr><td>L-BFGS max iterations per step</td><td>1, 5, 10, 20, or 30</td></tr><tr><td>L-BFGS learning rate</td><td>0.5 or 1.0</td></tr><tr><td>GN/LM refinement steps</td><td>25, 50, 75, 100, 125, 150, or 200</td></tr><tr><td>GN/LM initial damping</td><td> $5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , \mathrm { o r ~ 2 } \times 1 0 ^ { - 3 }$ </td></tr><tr><td>GN/LM Maximum backtracking steps</td><td>2, 4, 6, 8, and 10</td></tr><tr><td>GN/LM Damping shrink factor</td><td>Values in [0.4, 1.1]</td></tr><tr><td>GN/LM Damping growth factor</td><td>5, 10, and 15</td></tr><tr><td>Selection criterion</td><td>Smallest  $L ^ { 2 }$  error, with runtime used to break near-ties</td></tr></table>

Table 18. Nehari-constrained Deep Ritz architecture, constraint, and optimization hyperparameters tested for solving (10) on r0, 30s.

Table 18 lists the architecture, quadrature, constraint, and optimizer parameters tested in the hyperparameter sweep. The sweep varied the activation function, network depth and width, number of quadrature points, Adam warm-up length, L–BFGS settings, GN/LM refinement parameters, and the Pokhozhaev penalty weight $\lambda _ { \mathrm { { P o k h } } }$ . Since the Nehari projection enforces the first identity directly and the

hard trial function enforces the boundary conditions, the remaining penalty is used only to control the finite-interval second Pokhozhaev residual.
<table><tr><td colspan="3">Optimization Refinement Activation Layers per Layer</td><td colspan="3">Hidden Neurons</td><td rowspan="2">Adam Warm-up</td><td rowspan="2">GN/LM Steps</td><td rowspan="2">Max Backtracks</td><td rowspan="2">Damping</td><td rowspan="2">Damping Growth</td><td colspan="3">Initial Residual</td><td rowspan="2">Quotient  $\underline { { \lambda _ { \mathrm { P o k h } } } }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td> $N _ { q }$ </td><td>Shrink</td><td>Damping Points</td><td>Points</td></tr><tr><td>Adam</td><td>GN/LM</td><td>tanh(x)</td><td>5</td><td>64</td><td>29</td><td>10,000</td><td>200</td><td>6</td><td>0.7</td><td>15</td><td>10-3</td><td>128</td><td>128</td><td>0.1</td></tr></table>

Table 19. Selected Nehari-constrained Deep Ritz configuration for mixed boundary conditions (17).

The final configuration selected from this sweep is reported in Table 19. This baseline uses Nehari constrained Adam training followed by GN/LM residual refinement, with the quadrature and residualpoint counts fixed to balance accuracy, runtime, and memory usage. All Nehari-constrained Deep Ritz results below use this fixed configuration unless otherwise stated, so the following tables measure the performance of the selected Nehari baseline across the nonlinear powers rather than a separate hyperparameter search for each value of p.

Table 20 reports the primary-device results for the Nehari-constrained Deep Ritz method. Unlike the constrained Deep Ritz formulation above, this method does not prescribe the exact peak amplitude $Q ( 0 ) = A$ . Instead, the amplitude is determined dynamically through the Nehari projection (50), while the hard trial function enforces positivity, symmetry at the origin, and the endpoint condition at $x =$ L. Thus, this formulation tests whether the ground-state branch can be selected variationally without inserting the known exact amplitude.

<table><tr><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1> ${ \mathcal { L } } _ { \mathrm { N D R } } ( \theta )$ </td><td rowspan=1 colspan=1> ${ \mathcal { I } } _ { \mathrm { G N } } ( \theta )$ </td><td rowspan=1 colspan=1>CPU time</td><td rowspan=1 colspan=1> $L ^ { \infty }$ error</td><td rowspan=1 colspan=1> $\overline { { L ^ { 2 } } }$ error</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>6.666667e-01</td><td rowspan=1 colspan=1>2.141298e-13</td><td rowspan=1 colspan=1>313.008810</td><td rowspan=1 colspan=1>1.690701e-07</td><td rowspan=1 colspan=1>2.733932e-07</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6.801748e-01</td><td rowspan=1 colspan=1>1.853534e-11</td><td rowspan=1 colspan=1>315.139802</td><td rowspan=1 colspan=1>5.223740e-06</td><td rowspan=1 colspan=1>4.561398e-06</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6.677478e-01</td><td rowspan=1 colspan=1>1.803535e-09</td><td rowspan=1 colspan=1>300.718944</td><td rowspan=1 colspan=1>2.275145e-04</td><td rowspan=1 colspan=1>1.384767e-04</td></tr></table>

Table 20. Performance of the Nehari-constrained Deep Ritz method with mixed boundary conditions (17) for $Q ^ { \prime \prime } = Q - Q ^ { p }$

We observe that the Nehari projection gives a more stable approximation across the nonlinear powers tested. The method achieves $L ^ { 2 }$ errors of orders $1 0 ^ { - 7 } , 1 0 ^ { - 6 }$ , and $1 0 ^ { - 4 }$ for $p = 3 , 5$ , and 7, respectively. The higher-power case remains the most dificult, but the deterioration from $p = 5$ to $p = 7$ is much less severe than that observed for the constrained Deep Ritz method in Table 16. In particular, for $p = 7$ , the Nehari-constrained method reduces the $L ^ { 2 }$ error from order $1 0 ^ { - 3 }$ to $1 0 ^ { - 4 }$ . We interpret this as evidence that the Nehari projection provides a more efective mechanism for selecting the correct branch when the profile becomes sharper.

More specifically, Table 20 shows that, for $p = 3$ , both the $L ^ { \infty }$ and $L ^ { 2 }$ errors are of order $1 0 ^ { - 7 }$ . For $p = 5$ , both errors are of order $1 0 ^ { - 6 }$ , while for $p = 7$ they are of order $1 0 ^ { - 4 }$ . Thus, we obtain a gradual deterioration in profile accuracy as p increases, rather than the larger loss of accuracy observed for the constrained Deep Ritz formulation at $p = 7$

The improvement in robustness comes with a moderate increase in runtime. On the primary device, the Nehari-constrained runs require approximately 301–315 seconds, compared with approximately $2 4 7 -$ 264 seconds for the constrained Deep Ritz method. Thus, the Nehari formulation is not the fastest Deep Ritz variant, but we find that it gives a better balance of accuracy across all three powers.

In the left panel of Figure 8, one can observe that the Adam stage stabilizes the Nehari-constrained variational objective, after which the GN/LM stage further reduces the residual-refinement objective. In the right panel of the same figure, we compare the learned profiles with the exact ground state given by (14). The learned profiles are approaching the exact curve throughout the plotted interval, including near

![](images/160e7623afaf76e2e77fd36d99be982114645cdd34923997923920716fea2b2d.jpg)

$$
Q ^ { \prime \prime } { = } Q - Q ^ { 5 }
$$

![](images/15de1d11488d9b2d4c97a58f62dbbe47d6245596c4bbb1dc1c17d5ecf2bb2966.jpg)  
Figure 8. Training objective (left) and ground state solution profile convergence (right) for the Nehari-constrained Deep Ritz method (with parameters as seen in Table 19) for $Q ^ { \prime \prime } = Q - Q ^ { 5 }$ . The (black dashed) curve labeled Exact in the right panel is the explicit solution given by (14). The final learned profile corresponds to 10,000 Adam steps followed by 200 GN/LM steps.

the peak at the left endpoint, even though the exact peak amplitude is not prescribed by the formulation. The final profile after 10,000 Adam steps and 200 GN/LM steps is not only visually indistinguishable from the exact solution at the plotting scale, but also its remaining numerical discrepancy is consistent with the $p = 5$ errors reported in Table 20: the final $L ^ { \infty }$ and $L ^ { 2 }$ errors are $5 . 2 2 3 7 4 0 \times 1 0 ^ { - 6 }$ and $4 . 5 6 1 3 9 8 \times 1 0 ^ { - 6 }$ respectively, and are therefore, both of order $1 0 ^ { - 6 }$

4.2.1. Comparison with the “secondary” computational device for Nehari-constrained Deep Ritz. Table 21 shows that the Nehari-constrained Deep Ritz method benefits substantially from the secondary GPU+CPU workstation. On the primary device, the three runs require approximately 301–315 seconds, while on the secondary device the runtime is reduced to approximately 95–97 seconds. We find that this corresponds to a speedup of about three across the powers tested, with very little variation between $p = 3 , p = 5$ , and $p = 7$ . Thus, we observe that the secondary device provides a consistent acceleration rather than an improvement limited to a single exponent.

<table><tr><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1> ${ \mathcal { L } } _ { \mathrm { N D R } } ( \theta )$ </td><td rowspan=1 colspan=1> ${ \mathcal { I } } _ { \mathrm { G N } } ( \theta )$ </td><td rowspan=1 colspan=1>GPU+CPU time</td><td rowspan=1 colspan=1> $L ^ { \infty }$ error</td><td rowspan=1 colspan=1> $L ^ { 2 }$ error</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>6.666667e-01</td><td rowspan=1 colspan=1>5.934479e-13</td><td rowspan=1 colspan=1>96.631522</td><td rowspan=1 colspan=1>3.255530e-07</td><td rowspan=1 colspan=1>5.289161e-07</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6.801748e-01</td><td rowspan=1 colspan=1>1.871966e-11</td><td rowspan=1 colspan=1>94.922791</td><td rowspan=1 colspan=1>5.221092e-06</td><td rowspan=1 colspan=1>4.573099e-06</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6.677480e-01</td><td rowspan=1 colspan=1>2.051331e-09</td><td rowspan=1 colspan=1>96.848960</td><td rowspan=1 colspan=1>2.730569e-04</td><td rowspan=1 colspan=1>1.542854e-04</td></tr></table>

Table 21. Secondary-device GPU+CPU performance of the Nehari-constrained Deep Ritz method with mixed boundary conditions (17) for $Q ^ { \prime \prime } = Q - Q ^ { p }$

Importantly, this reduction in runtime does not substantially change the accuracy pattern. The final errors remain close to the primary-device values and preserve the same ordering across the nonlinear powers: the method is most accurate for $p = 3$ , remains highly accurate for $p = 5$ , and is least accurate for the sharper $p = 7$ profile. For $p = 3 .$ both the secondary-device errors remain of order $1 0 ^ { - 7 }$ ; for $p = 5$ both remain of order $1 0 ^ { - 6 } ;$ and for $p = 7$ , both remain of order $1 0 ^ { - 4 }$ . Thus, we observe exactly the same error orders as on the primary device. We find that the main efect of moving the Nehari-constrained Deep Ritz method to the secondary device is a roughly threefold reduction in wall-clock time, while the numerical accuracy remains essentially comparable.

The close agreement between the two devices is also reflected in the residual-refinement objective. The values of ${ \mathcal { I } } _ { \mathrm { G N } } ( \theta )$ remain of the same order on both devices for every power tested: $1 0 ^ { - 1 3 }$ for $p = 3$ 2 $1 0 ^ { - 1 1 }$ for $p = 5$ , and $1 0 ^ { - 9 }$ for $p = 7$ , difering only in the leading constant. Thus, unlike pure $\mathrm { G N / L M }$ in Section 3.1.3, the Nehari-constrained formulation does not exhibit a device-dependent change of solution branch.

4.3. Weinstein Deep Ritz Formulation. We next introduce a variational neural-network method based on the Weinstein quotient rather than on the action. Following the classical variational characterization [50], we use the Weinstein quotient to define the best constant as

$$
R _ { b } ( p ) : = \operatorname* { i n f } _ { \substack { u \in H ^ { 1 } ( \mathbb { R } ) \backslash \{ 0 \} } } \frac { q _ { b } ( u ) } { \| u \| _ { L ^ { p + 1 } ( \mathbb { R } ) } ^ { 2 } } ,\tag{56}
$$

where the quadratic form $q$ is defined as

$$
q _ { b } ( u ) : = \| u ^ { \prime } \| _ { L ^ { 2 } ( \mathbb { R } ) } ^ { 2 } + b \| u \| _ { L ^ { 2 } ( \mathbb { R } ) } ^ { 2 } .\tag{57}
$$

For brevity, we drop the dependence of $R _ { b }$ on $p ,$ and simply write $R _ { b }$ . If v is a positive optimizer normalized by $\| v \| _ { L ^ { p + 1 } ( \mathbb { R } ) } = 1$ , then $q _ { b } ( v ) = R _ { b }$ , and the corresponding Euler–Lagrange equation is

$$
- v ^ { \prime \prime } + b v = R _ { b } v ^ { p } .\tag{58}
$$

Consequently,

$$
Q = \left( { \frac { R _ { b } } { \gamma } } \right) ^ { \frac { 1 } { p - 1 } } v\tag{59}
$$

solves the initial ground state equation (11).

This characterization is especially convenient for a Deep Ritz implementation. The quotient is invariant under multiplication of u by a nonzero constant, so it does not select the trivial solution. We remove the remaining scale invariance by normalizing the neural-network trial function exactly. The optimization determines the normalized shape $v$ and the value $R _ { b } ;$ the physical amplitude of $Q$ is then recovered from (59). In particular, for fixed b and $p ,$ the variational training is independent of $\gamma ,$ which enters only through the final amplitude rescaling.

Since the ground state is even, we work on $[ 0 , L ]$ and interpret each trial function by its even extension to $[ - L , L ]$ . Let $q _ { \theta }$ be a neural network and let $\rho : \mathbb { R } \to ( 0 , \infty )$ be a smooth positivity map, for example, $\rho ( z ) = \log ( 1 + e ^ { z } )$ . We use the hard-constrained raw trial function

$$
w _ { \theta } ( x ) = \left( 1 - { \frac { x ^ { 2 } } { L ^ { 2 } } } \right) \rho \left( q _ { \theta } \left( { \frac { x ^ { 2 } } { L ^ { 2 } } } \right) \right) , \qquad 0 \leqslant x \leqslant L ,\tag{60}
$$

which gives

$$
w _ { \theta } ^ { \prime } ( 0 ) = 0 , \qquad w _ { \theta } ( L ) = 0 , \qquad w _ { \theta } ( x ) > 0 \quad \mathrm { f o r ~ } 0 \leqslant x < L .
$$

We normalize the even extension of w<sub>θ</sub> by defining

$$
v _ { \theta } ( x ) : = A _ { \theta } ^ { - \frac { 1 } { p + 1 } } w _ { \theta } ( x ) , \mathrm { ~ w h e r e ~ } A _ { \theta } = 2 \int _ { 0 } ^ { L } | w _ { \theta } ( x ) | ^ { p + 1 } d x .\tag{61}
$$

We then have $2 \int _ { 0 } ^ { L } | v _ { \theta } ( x ) | ^ { p + 1 } d x = 1$

The truncated Weinstein quotient, evaluated on the even extension, is

$$
R _ { b , L } ( \theta ) : = 2 \int _ { 0 } ^ { L } \left( | v _ { \theta } ^ { \prime } ( x ) | ^ { 2 } + b | v _ { \theta } ( x ) | ^ { 2 } \right) d x = \frac { 2 \sum _ { 0 } ^ { L } \left( | w _ { \theta } ^ { \prime } ( x ) | ^ { 2 } + b | w _ { \theta } ( x ) | ^ { 2 } \right) d x } { \left( 2 \sum _ { 0 } ^ { L } | w _ { \theta } ( x ) | ^ { p + 1 } d x \right) ^ { \frac { 2 } { p + 1 } } } .\tag{62}
$$

The Weinstein Deep Ritz loss is then defined as

$$
{ \mathcal { L } } _ { \mathrm { W D R } } ( \theta ) = R _ { b , L } ( \theta ) .\tag{63}
$$

We point out that no other normalization penalty, amplitude condition, or Dirichlet boundary penalty is needed because we built these constraints into (60)–(61).

After minimizing (63), following the definition of Q in (59), we set

$$
Q _ { \theta } ( x ) = \left( \frac { R _ { b , L } ( \theta ) } { \gamma } \right) ^ { \frac { 1 } { p - 1 } } v _ { \theta } ( x ) .\tag{64}
$$

At an exact constrained minimizer (i.e., a minimizer of the underlying function space variational problem, rather than a minimizer over a fixed neural-network class), the normalized profile satisfies

$$
\begin{array} { r } { - v _ { \theta } ^ { \prime \prime } + b v _ { \theta } = R _ { b , L } ( \theta ) v _ { \theta } ^ { p } , \qquad v _ { \theta } ^ { \prime } ( 0 ) = 0 , \qquad v _ { \theta } ( L ) = 0 , } \end{array}\tag{65}
$$

and also $- Q _ { \theta } ^ { \prime \prime } + b Q _ { \theta } - \gamma Q _ { \theta } ^ { p } = 0 .$

During and after training, we monitor the Euler–Lagrange residual

$$
r _ { v , \theta } ( x ) = - v _ { \theta } ^ { \prime \prime } ( x ) + b v _ { \theta } ( x ) - R _ { b , L } ( \theta ) ( v _ { \theta } ( x ) ) ^ { p } .\tag{66}
$$

The corresponding profile residual satisfies

$$
- Q _ { \theta } ^ { \prime \prime } + b Q _ { \theta } - \gamma Q _ { \theta } ^ { p } = \left( \frac { R _ { b , L } ( \theta ) } { \gamma } \right) ^ { \frac { 1 } { p - 1 } } r _ { v , \theta } .\tag{67}
$$

Thus, either residual (66) or (67) may be used as a post-training diagnostic. If a GN/LM stage is used, it is applied only after the Weinstein-quotient pretraining to a least-squares vector assembled from ${ r } _ { v , \theta }$ at residual points. The normalization and the hard boundary structure are retained during this refinement, so that the optimization cannot collapse to the zero solution.

Unlike the action-based Deep Ritz formulation, the Weinstein Deep Ritz method does not require the value $Q ( 0 ) = Q _ { 0 }$ as input. It obtains the normalized profile shape from the quotient minimization and then recovers the correct amplitude through (64). Thus, forming the WDR loss and reconstruction does not require an explicit profile, which makes this approach promising for computing ground states with no explicit solutions, which, for example, we investigate in the follow-up work [14]. The Weinstein Deep Ritz implementation is summarized in Algorithm 5.

The method first trains a normalized profile by minimizing the truncated Weinstein quotient and then reconstructs the physical ground-state amplitude from the computed quotient value.

For the GN/LM refinement, the residual points are uniformly spaced on r0, Ls and the residual vector is formed using the square-root composite-trapezoidal weights,

$$
F _ { \mathrm { G N } , j } ( \theta ) = \sqrt { \omega _ { j } } r _ { v , \theta } ( \widehat { x } _ { j } ) , \qquad j = 1 , \ldots , N _ { r } ,
$$

where $\omega _ { 1 } = \omega _ { N _ { r } } = \Delta x / 2 , \omega _ { j } = \Delta x$ for $2 \leqslant j \leqslant N _ { r } - 1$ , and $\Delta x = L / ( N _ { r } - 1 )$ . No additional $N _ { r } ^ { - 1 / 2 }$ normalization is used, unlike in the PINN formulation. The general GN/LM damping and backtracking procedure is the one described above. During the variational stage, $N _ { q }$ denotes the fixed quadrature grid used to evaluate the Weinstein quotient. During GN/LM refinement, the residual grid is used to form $F _ { \mathrm { G N } }$ , while a separate quotient grid is used to recompute the normalization and $R _ { b , L } ( \theta )$ at each trial parameter vector.

Thus, the hyperparameter sweep in Table 22 varies both the neural-network approximation parameters and the optimization parameters that afect the quotient minimization and subsequent residual refinements. In particular, we varied the activation function, network depth and width, number of quadrature points used to evaluate the quotient, Adam warm-up length, L–BFGS settings, and GN/LM refinement parameters. Since the normalization and boundary structure are built into the Weinstein trial function, no separate amplitude penalty, normalization penalty, or boundary penalty is required.

The selected configuration is reported in Table 23. This baseline uses an Adam-trained Weinstein quotient stage followed by GN/LM residual refinement, with the quadrature and residual-point counts fixed in order to balance accuracy with memory usage and runtime. All Weinstein Deep Ritz results reported below use this fixed configuration unless otherwise stated, so the subsequent tables measure the performance of the selected Weinstein baseline across the nonlinear powers rather than a new hyperparameter search for each case.

```latex
Algorithm 5 Weinstein Deep Ritz Method for $Q ^ { \prime \prime } = b Q - \gamma Q ^ { p }$
1 Choose $b > 0 , \gamma > 0 , p > 1 ,$ a truncated domain $[ 0 , L ]$ , quadrature points $\{ x _ { j } \} _ { j = 1 } ^ { N _ { q } }$ , and optional residual-refinement points
$\{ \widehat { x } _ { j } \} _ { j = 1 } ^ { N _ { r } } .$
2 Define a neural network $q _ { \theta }$ and a smooth positivity map $\rho : \mathbb { R } \to ( 0 , \infty )$ such that
$\rho ( s ) = \log ( 1 + e ^ { s } ) .$
3 Define the hard-constrained raw trial function
$w _ { \theta } ( x ) = \left( 1 - \frac { x ^ { 2 } } { L ^ { 2 } } \right) \rho \bigg ( q _ { \theta } \bigg ( \frac { x ^ { 2 } } { L ^ { 2 } } \bigg ) \bigg )$
which satisfies $w _ { \theta } ^ { \prime } ( 0 ) = 0$ and $w _ { \theta } ( L ) = 0 .$
4 Use automatic diferentiation to compute $w _ { \theta } ^ { \prime } ( x _ { j } ) .$
5 Compute the normalization
$A _ { \theta } = 2 \int _ { 0 } ^ { L } | w _ { \theta } | ^ { p + 1 } d x , \qquad v _ { \theta } = A _ { \theta } ^ { - 1 / ( p + 1 ) } w _ { \theta } .$
6 Compute the truncated Weinstein quotient
$R _ { b , L } ( \theta ) = 2 \int _ { 0 } ^ { L } \left( | v _ { \theta } ^ { \prime } | ^ { 2 } + b | v _ { \theta } | ^ { 2 } \right) d x .$
7 Define the variational loss
$\mathcal { L } _ { \mathrm { W D R } } ( \theta ) = R _ { b , L } ( \theta ) .$
8 Choose a quotient-minimization strategy
Opt P tAdam, L–BFGS, Adam Ñ L–BFGSu.
9 for $n = 1 , 2 , \ldots , N _ { \mathrm { e p o c h s } }$ do
10 Evaluate the quadrature approximations of $A _ { \theta }$ and $R _ { b , L } ( \theta )$ on the fixed uniform quadrature grid.
11 Diferentiate ${ \mathcal { L } } _ { \mathrm { W D R } }$ with respect to θ and update θ using the variational stage of Opt .
12 Store $R _ { b , L } ( \theta )$ together with the normalization and boundary diagnostics.
13 end for
14 if GN/LM refinement is selected then
15 Recompute $A _ { \theta } , v _ { \theta } ,$ and $R _ { b , L } ( \theta )$ and form the Euler–Lagrange residual
$r _ { v , \theta } ( x ) = - v _ { \theta } ^ { \prime \prime } ( x ) + b v _ { \theta } ( x ) - R _ { b , L } ( \theta ) ( v _ { \theta } ( x ) ) ^ { p }$
16 Form $F _ { \mathrm { G N } } ( \boldsymbol { \theta } )$ from the quadrature-weighted values of ${ r } _ { v , \theta }$ at the GN/LM residual points.
17 Apply GN/LM to minimize
$\mathcal { T } _ { \mathrm { G N } } ( \theta ) = \frac { 1 } { 2 } \| F _ { \mathrm { G N } } ( \theta ) \| _ { 2 } ^ { 2 } ,$
while retaining the normalized, hard-constrained representation of $v _ { \theta }$
18 end if
19 Reconstruct the ground-state profile
$Q _ { \theta } = \left( \frac { R _ { b , L } ( \theta ) } { \gamma } \right) ^ { 1 / ( p - 1 ) } v _ { \theta } .$
20 Evaluate the pointwise ground-state residual and the mass, energy, and finite-interval Pokhozhaev diagnostics for $Q _ { \theta }$
21 Return $v _ { \theta } , R _ { b , L } ( \theta )$ , and the reconstructed profile $Q _ { \theta }$
```

Table 24 reports the performance of the Weinstein Deep Ritz method for $Q ^ { \prime \prime } = Q - Q ^ { p }$ , with $p = 3 , 5$ and 7. Unlike the action-based Deep Ritz and Nehari-constrained formulations, this method does not minimize the action directly. Instead, the Adam stage minimizes the Weinstein quotient ${ \mathcal { L } } _ { \mathrm { W D R } } ( \theta )$ , which selects a normalized profile shape. The physical amplitude is then recovered from the computed quotient value using the reconstruction formula (64). As before, ${ \mathcal { I } } _ { \mathrm { G N } } ( \theta )$ denotes the separate residual-based GN/LM refinement objective, not the quotient itself.

We find that the selected Weinstein Deep Ritz configuration gives the most favorable error–runtime balance among the Deep Ritz variants tested. For the cubic case, the method achieves errors of order $1 0 ^ { - 8 }$ while for the quintic case the errors are of order $1 0 ^ { - 6 } .$ . The septic case remains the most challenging, with errors increasing to the $1 0 ^ { - 4 }$ level. Thus, we observe a gradual deterioration in accuracy as the profile becomes sharper, but the Weinstein formulation still remains accurate across all three nonlinear powers.

Parameter Values tested / Description   
Domain length L 30   
Boundary condition mixed   
Trial function Soft penalty trial, hard or decaying Dirichlet ansatz   
Optimizer Adam, L–BFGS, or Adam–L–BFGS   
Refinement strategy GN/LM or Adam–GN/LM   
Activation functions $\operatorname { t a n h } ( x ) , \operatorname { S i L U } ( x ) , \operatorname { S i L U P L U S } ( x ) , e ^ { - x ^ { 2 } } , e ^ { - x ^ { 4 } }$ , and sechpxq   
Number of hidden layers 3, 4, or 5   
Neurons per layer 32, 64, or 96   
Number of quadrature points $N _ { q }$ $\phantom { - } 2 ^ { 8 } , 2 ^ { 9 } , 2 ^ { 1 0 } , 2 ^ { 1 1 } , \phantom { - } \mathrm { o r } 2 ^ { 1 2 }$   
Adam warm-up length 2,500, 5,000, 7,500, 10,000, 12,500, or 15,000 steps   
Adam learning rate lr 10<sup>´3</sup> initial; StepLR decay   
L–BFGS refinement steps 25, 50, 75, 100, 125, 150, or 200   
L–BFGS history size 20, 50, or 100   
L–BFGS max iterations per step 1, 5, 10, 20, or 30   
L–BFGS learning rate 0.5 or 1.0   
GN/LM refinement steps 25, 50, 75, 100, 125, 150, or 200   
GN/LM initial damping $5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , \mathrm { o r ~ 2 } \times 1 0 ^ { - 3 }$   
GN/LM Maximum backtracking steps 2, 4, 6, 8, and 10   
GN/LM Damping shrink factor Values in r0.4, 1.1s   
GN/LM Damping growth factor 5, 10, and 15   
Selection criterion Smallest $L ^ { 2 }$ error, with runtime used to break near-ties

Table 22. Weinstein Deep Ritz architecture, constraint, and optimization hyperparameters tested for solving (10) on r0, 30s.
<table><tr><td colspan="4">Optimization Refinement Activation</td><td colspan="2">Hidden Neurons</td><td colspan="2">Adam</td><td colspan="2">GN/LM GN/LM Max Damping</td><td colspan="2">Damping Growth</td><td colspan="2">Initial Residual Quotient</td></tr><tr><td></td><td></td><td></td><td>Layers</td><td>per Layer</td><td> $N _ { q }$ </td><td>Warm-up</td><td>Steps</td><td>Backtracks</td><td>Shrink</td><td>Damping</td><td>Points</td><td>Points</td></tr><tr><td>Adam</td><td>GN/LM</td><td>SiLU(x)</td><td>5</td><td>32</td><td>29</td><td>5,000</td><td>200</td><td>6</td><td>0.5</td><td>15  $\overline { { 1 0 ^ { - 3 } } }$ </td><td>128</td><td>128</td></tr></table>

Table 23. Selected Weinstein Deep Ritz configurations for mixed boundary conditions 17.

<table><tr><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1> ${ \mathcal { L } } _ { \mathrm { W D R } } ( \theta )$ </td><td rowspan=1 colspan=1> ${ \mathcal { I } } _ { \mathrm { G N } } ( \theta )$ </td><td rowspan=1 colspan=1>CPU time</td><td rowspan=1 colspan=1> $L ^ { \infty }$ error</td><td rowspan=1 colspan=1> $L ^ { 2 }$ error</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>2.309401e+00</td><td rowspan=1 colspan=1>5.781202e-15</td><td rowspan=1 colspan=1>148.970724</td><td rowspan=1 colspan=1>4.832268e-08</td><td rowspan=1 colspan=1>6.092994e-08</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>2.553766e+00</td><td rowspan=1 colspan=1>3.953484e-14</td><td rowspan=1 colspan=1>150.616751</td><td rowspan=1 colspan=1>5.235757e-06</td><td rowspan=1 colspan=1>2.322385e-06</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>2.592439e+00</td><td rowspan=1 colspan=1>1.687038e-09</td><td rowspan=1 colspan=1>145.797130</td><td rowspan=1 colspan=1>2.802301e-04</td><td rowspan=1 colspan=1>1.451584e-04</td></tr></table>

Table 24. Performance of the Weinstein Deep Ritz method with mixed boundary conditions (17) for $Q ^ { \prime \prime } = Q - Q ^ { p }$

More precisely, Table 24 shows that both errors are of order $1 0 ^ { - 8 }$ for $p = 3$ and order $1 0 ^ { - 6 }$ for $p = 5$ For $p = 7 .$ , both errors increase to order $1 0 ^ { - 4 }$ . Thus, we obtain the same gradual loss of accuracy with increasing p as in the Nehari-constrained formulation, but without the order $1 0 ^ { - 3 }$ error observed for the action-based constrained Deep Ritz method.

The runtime is also substantially lower than for the other Deep Ritz variants. The Weinstein runs re quire approximately 146–151 seconds, whereas the constrained Deep Ritz method requires approximately 247–264 seconds and the Nehari-constrained method requires approximately 301–315 seconds. Hence, among the selected configurations, the Weinstein formulation is the most accurate Deep Ritz variant for $p = 3$ and the fastest for all three powers. We observe that for $p = 5$ , the Nehari-constrained formulation gives the slightly smaller $L ^ { \infty }$ error, while the Weinstein formulation gives the smaller $L ^ { 2 }$ . For $p = 7$ , its errors are comparable to those of the Nehari-constrained method, with both methods producing errors of order $1 0 ^ { - 4 }$ , while the Weinstein method retains the shorter runtime. Hence, we find that the Weinstein formulation gives the best overall balance between profile accuracy and computational cost among the three selected Deep Ritz variants we consider in this paper.

![](images/065adb46137c59a18a830059ca177e050891886d1a8a8925e9dcfb4efdf9f872.jpg)

![](images/ef868f11c30983ed2bf05c62d77a29d6fae12242a6cc09e7625b0ad69ba97c94.jpg)  
Figure 9. Training objective (left) and ground state solution profile convergence $( \mathrm { r i g h t } )$ for the Weinstein Deep Ritz method (with parameters as seen in Table 23) for $Q ^ { \prime \prime } = Q { - } Q ^ { 5 }$ The (black dashed) curve labeled Exact in the right panel is the explicit solution given by (14). The final learned profile corresponds to 5,000 Adam steps followed by 200 GN/LM steps.

Figure 9 illustrates the two-stage behavior of the method. In the left panel, we observe that the Adam warm-up reduces the Weinstein quotient, thereby improving the normalized profile shape and quotient value. The GN/LM stage then acts as a residual-refinement step and sharply reduces the residual-based objective. In the right panel of Figure 9, we compare the reconstructed profiles with the exact ground state given by (14). We observe that the Adam-trained approximation already captures the main shape and amplitude of the ground state, while the GN/LM stage further improves the agreement with the exact profile. This behavior is consistent with the structure of the method: the variational stage determines the normalized shape, while the final amplitude is obtained only after the quotient minimization. The final profile after 5,000 Adam steps and 200 GN/LM steps is visually indistinguishable from the exact solution at the plotting scale. Its remaining discrepancy agrees with the $p = 5$ values in Table 24: the final $L ^ { \infty }$ and $L ^ { 2 }$ errors are $5 . 2 3 5 7 5 7 \times 1 0 ^ { - 6 }$ and $2 . 3 2 2 3 8 5 \times 1 0 ^ { - 6 }$ , respectively, and are hence, both of order $1 0 ^ { - 6 }$ . Although the computations are performed on the full interval r0, 30s, the solution profiles are displayed on r0, 10s to emphasize the left side of the profile and the peak-amplitude region.

4.3.1. Comparison with the “secondary” computational device for Weinstein Deep Ritz. Table 25 reports the same Weinstein Deep Ritz baseline on the secondary GPU+CPU workstation. We observe that the main efect of the secondary device is a reduction in runtime from approximately 146–151 seconds to approximately 40–44 seconds. This corresponds to a speedup of about 3.3–3.7 across the three nonlinear powers. The acceleration is consistent across $p = 3 , p = 5$ , and $p = 7$ , indicating that the secondary device improves the practical wall-clock cost without changing the qualitative behavior of the method.

The secondary-device errors remain of the same orders as the corresponding primary-device results. For $p = 3 ,$ the $L ^ { \infty }$ error remains of order $1 0 ^ { - 8 }$ , while the $L ^ { 2 }$ error changes from order $1 0 ^ { - 8 }$ to order $1 0 ^ { - 7 }$ . For $p = 5$ , both errors remain of order $1 0 ^ { - 6 }$ , and for $p = 7$ , both remain of order $1 0 ^ { - 4 }$ . Thus, we find that the GPU+CPU implementation mainly improves runtime while preserving the same accuracy pattern. In particular, the higher-power case remains the most dificult, but the secondary-device result stays comparable to the original Weinstein run.

<table><tr><td rowspan=1 colspan=1>p</td><td rowspan=1 colspan=1> ${ \mathcal { L } } _ { \mathrm { W D R } } ( \theta )$ </td><td rowspan=1 colspan=1> ${ \mathcal { I } } _ { \mathrm { G N } } ( \theta )$ </td><td rowspan=1 colspan=1>GPU+CPU time</td><td rowspan=1 colspan=1> $L ^ { \infty }$ error</td><td rowspan=1 colspan=1> $L ^ { 2 }$ error</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1> $2 . 3 0 9 4 0 1 \mathrm { e } { + } 0 0$ </td><td rowspan=1 colspan=1>1.934108e-14</td><td rowspan=1 colspan=1>40.420787</td><td rowspan=1 colspan=1>9.883258e-08</td><td rowspan=1 colspan=1>1.186556e-07</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1> $2 . 5 5 3 7 6 6 \mathrm { e } + 0 0$ </td><td rowspan=1 colspan=1>5.054238e-14</td><td rowspan=1 colspan=1>44.468209</td><td rowspan=1 colspan=1>5.204065e-06</td><td rowspan=1 colspan=1>2.321890e-06</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1> $2 . 5 9 2 4 3 9 \mathrm { { e + 0 0 } }$ </td><td rowspan=1 colspan=1>1.258231e-09</td><td rowspan=1 colspan=1>44.377295</td><td rowspan=1 colspan=1>3.016073e-04</td><td rowspan=1 colspan=1>1.542014e-04</td></tr></table>

Table 25. Secondary-device GPU+CPU performance of the Weinstein Deep Ritz method with mixed boundary conditions (17) for $Q ^ { \prime \prime } = Q - Q ^ { p }$

The quotient values are also essentially unchanged between the two devices for all three powers. Similarly, the primary- and secondary-device residual-refinement objectives remain of the same orders for p “ 5 and p “ 7. For $p = 3 , { \mathcal { T } } _ { \mathrm { G N } } ( \theta )$ changes from order 10<sup>´15</sup> to order 10<sup>´14</sup>, while the corresponding profile errors remain very small. Thus, we observe no evidence of a device-dependent change in the selected solution branch.

Overall, we find that the Weinstein Deep Ritz method gives the best error–runtime combination among the Deep Ritz variants considered here for the selected runs. It is the fastest formulation and gives the smallest errors for $p = 3$ , while remaining comparable to the Nehari-constrained formulation for $p = 5$ and $p = 7$ . The Nehari-constrained Deep Ritz method, however, remains competitive for the sharper $p = 7$ profile and selects the same solution branch in the reported runs without prescribing the exact peak amplitude. Thus, we regard the Weinstein formulation as the most eficient and often the most accurate variant, among the selected runs, while Nehari provides a robust constrained alternative whose accuracy remains comparable for the higher-power case.

## 5. Conclusions and Discussion

We have developed and compared enhanced neural-network-based methods for computing one-dimensional ground-state profiles for several classical models in nonlinear dispersive equations. The profile equation $Q ^ { \prime \prime } = b Q - \gamma Q ^ { p }$ , arising from the NLS, gKdV, and NLKG models, admits explicit sech-type solutions for all integer power $p > 1$ , thus, providing exact benchmarks for systematic error analysis. Building on the comparison in [15], we investigated three directions of improvement over the standard Adam-trained PINN baseline: (i) higher-order optimization (GN/LM refinement), (ii) Fourier-feature PINNs, and (iii) constrained Deep Ritz formulations.

The main findings of this study are as follows.

Higher-order optimization improves PINN accuracy. Replacing or supplementing standard Adam training with Gauss–Newton/Levenberg–Marquardt refinement consistently reduces the pointwise and $L ^ { 2 }$ errors of standard PINNs. An Adam warm-up followed by GN/LM refinement (Adam–GN/LM) is generally more robust than pure GN/LM, which is sensitive to the damping parameter and initialization and may converge to an incorrect solution branch, such as a double-bump profile. GN/LM refinement generally gives the best results for the standard PINN and Deep Ritz formulations, whereas L–BFGS performs better for the FF–PINN formulation.

Fourier-feature PINNs provide modest additional gains. Embedding the spatial coordinate into a trigonometric feature space before passing it through the multilayer perceptron is designed to reduce the spectral bias of coordinate-based networks and, in some configurations, we observe improved convergence. However, the gains from Fourier features are smaller than those from higher-order optimization, and the method introduces additional hyperparameters including the scale of the frequency matrix and whether the raw spatial coordinate is included.

Deep Ritz variants outperform residual-based methods. Among all neural-network methods tested, the variational Deep Ritz formulations produce the strongest overall performance. By minimizing the action or related variational objectives instead of a pointwise residual, these methods exploit the variational structure of the ground-state equation and yield lower errors for comparable or shorter runtimes.

The Weinstein Deep Ritz method gives the best error–runtime balance overall. The Weinstein Deep Ritz formulation, based on minimizing the Weinstein quotient (22) over a normalized, hard-constrained trial class, achieves the smallest errors among the tested neural-network methods for $p = 3 .$ , while also requiring the shortest runtimes, among the Deep Ritz variants (approximately 146–151 seconds on the primary device versus 247–264 seconds for the constrained Deep Ritz and 301–315 seconds for the Nehariconstrained variant). For $p = 5$ and $p = 7$ , the Weinstein and Nehari-constrained formulations produce errors on the same order $( \sim 1 0 ^ { - 6 } \ \mathrm { a n d } \sim 1 0 ^ { - 4 }$ , respectively). A key advantage of the Weinstein method is that it does not require prescribing the exact peak amplitude $Q ( 0 )$ ; the normalized profile shape is determined by the quotient minimization, and the physical amplitude is recovered through the reconstruction formula (24). This makes the method promising to problems where no explicit solution is available.

The Nehari-constrained Deep Ritz method provides a robust alternative. The Nehari-constrained formulation, which projects the amplitude onto the Nehari manifold and supplements the action with a Pokhozhaev penalty, provides a stable branch-selection mechanism without prescribing the exact peak amplitude. Its accuracy remains competitive, particularly for the sharper $p = 7$ profile.

The dependence of accuracy on the nonlinear power is method-dependent. The numerical results do not show a uniform deterioration in accuracy as p increases from 3 to 7. For the Deep Ritz formulations, the higher-power cases generally produce larger errors. For the PINN formulations, the trend depends on the optimizer: the pure L–BFGS errors are smaller at $p \ = \ 7$ than at $p \ = \ 3$ , the pure GN/LM errors decrease from $p = 3$ to $p = 5$ before increasing at $p = 7 ,$ and the Adam–L–BFGS and Adam– GN/LM configurations exhibit diferent trends across the tested powers. The FF–PINN errors remain of comparable order across the three powers. Thus, the observed dependence on the nonlinear power varies with the selected formulation and optimizer.

GPU acceleration preserves accuracy while reducing runtime. Moving from the primary CPU-based device to the secondary GPU+CPU workstation consistently reduces wall-clock runtimes, with primary to-secondary runtime ratios ranging from approximately 1.6 to 9.7 across the reported configurations, while the profile errors remain of comparable order for most of the reported configurations.

Neural-network methods do not yet match classical Petviashvili solvers. Although the enhanced neuralnetwork methods substantially improve upon the Adam-trained PINN baseline, they do not yet achieve in 1D the accuracy or computational eficiency of the Petviashvili iteration reported in [15]. For the onedimensional benchmark problem, the Petviashvili method achieves machine-precision accuracy, on the order of $1 0 ^ { - 1 4 }$ , in shorter runtimes. Further studies would be needed, for example, for higher dimensions and more complex nonlinearities.

The methods developed here provide a one-dimensional proof of concept, they are intended to establish a foundation for extending neural-network-based ground-state computation to settings where classical methods face greater challenges: higher-dimensional problems subject to the curse of dimensionality, equations with combined or non-standard nonlinearities where explicit solutions are unavailable, and problems involving fractional or higher-order diferential operators. The Weinstein Deep Ritz method, which we introduced in this paper, does not rely on knowledge of the exact solution and is therefore prospectively well-suited for such extensions.

## Appendix A. Newton direction and the strong Wolfe conditions

For reader’s convenience in this appendix we give a standard derivation of the quasi-Newton search direction (33) and the strong Wolfe conditions used in the L–BFGS refinement stage (though the material is by now classical, see, e.g., [33] for a full treatment).

Newton search direction. First, note that the Hessian $\nabla _ { { \boldsymbol { \theta } } } ^ { 2 } { \mathcal { L } } ( { \boldsymbol { \theta } } _ { k } )$ is the matrix of second derivatives of the loss at the given current iterate $\theta _ { k }$ , thus, it describes its local second-order behavior. The second-order Taylor expansion about $\theta _ { k }$ gives the quadratic expansion

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } _ { k } + \boldsymbol { p } ) \approx \mathcal { L } ( \boldsymbol { \theta } _ { k } ) + \nabla _ { \boldsymbol { \theta } } \mathcal { L } ( \boldsymbol { \theta } _ { k } ) ^ { T } \boldsymbol { p } + \frac { 1 } { 2 } \boldsymbol { p } ^ { T } \nabla _ { \boldsymbol { \theta } } ^ { 2 } \mathcal { L } ( \boldsymbol { \theta } _ { k } ) \boldsymbol { p } , } \end{array}\tag{68}
$$

whose stationary point $p _ { k }$ solves

$$
\nabla _ { \theta } ^ { 2 } \mathcal { L } ( \theta _ { k } ) p _ { k } = - \nabla _ { \theta } \mathcal { L } ( \theta _ { k } ) .\tag{69}
$$

If $\nabla _ { { \boldsymbol { \theta } } } ^ { 2 } { \mathcal { L } } ( { \boldsymbol { \theta } } _ { k } )$ is positive definite, this stationary point is the unique minimizer of the quadratic model. When $\nabla _ { { \boldsymbol { \theta } } } ^ { 2 } { \mathcal { L } } ( { \boldsymbol { \theta } } _ { k } )$ is non-singular, then $p _ { k } = - \big ( \nabla _ { \theta } ^ { 2 } \mathcal { L } ( \theta _ { k } ) \big ) ^ { - 1 } \nabla _ { \theta } \mathcal { L } ( \theta _ { k } )$ , see [33, Sec. 2.2, Eqs. (2.14)–(2.15)]. In practice, however, the search direction is obtained by solving (69) without forming the inverse. Forming and storing the full $N _ { \theta } \times N _ { \theta }$ Hessian is impractical for large networks, so L–BFGS instead uses a limited number of recent parameter and gradient diferences

$$
\begin{array} { r } { s _ { j } = \theta _ { j + 1 } - \theta _ { j } , \qquad y _ { j } = \nabla _ { \theta } \mathcal { L } ( \theta _ { j + 1 } ) - \nabla _ { \theta } \mathcal { L } ( \theta _ { j } ) . } \end{array}
$$

A first-order expansion of the gradient gives $y _ { j } \approx \nabla _ { \boldsymbol { \theta } } ^ { 2 } \mathcal { L } ( \boldsymbol { \theta } _ { j } ) s _ { j }$ , so gradient changes carry information about the action of the Hessian. Then the corresponding inverse secant relation is $H _ { j + 1 } y _ { j } \approx s _ { j }$ with $H _ { j + 1 } \approx \left( \nabla _ { \boldsymbol { \theta } } ^ { 2 } \mathcal { L } ( \boldsymbol { \theta } _ { j + 1 } ) \right) ^ { - 1 }$ , see [27, Sec. 2], [33, Sec. 7.2, Algs. 7.4–7.5] for more details. This yields the search direction (33).

Strong Wolfe conditions. Once $p _ { k }$ is fixed, a line search minimizes $\phi _ { k } ( \alpha ) = \mathcal { L } ( \theta _ { k } + \alpha p _ { k } )$ over $\alpha > 0$ Writing $g _ { k } = \nabla _ { \theta } \mathcal { L } ( \theta _ { k } )$ and $\phi _ { k } ^ { \prime } ( 0 ) = g _ { k } ^ { T } p _ { k }$ , and assuming $p _ { k }$ is a descent direction $( g _ { k } ^ { T } p _ { k } < 0 )$ , the strong Wolfe conditions require

$$
\mathcal { L } ( \theta _ { k } + \alpha _ { k } p _ { k } ) \leqslant \mathcal { L } ( \theta _ { k } ) + c _ { 1 } \alpha _ { k } g _ { k } ^ { T } p _ { k } ,\tag{70}
$$

$$
\begin{array} { r } { \big | \nabla _ { \theta } \mathcal { L } ( \theta _ { k } + \alpha _ { k } p _ { k } ) ^ { T } p _ { k } \big | \leqslant c _ { 2 } | g _ { k } ^ { T } p _ { k } | , \qquad 0 < c _ { 1 } < c _ { 2 } < 1 . } \end{array}\tag{71}
$$

Condition (70) is for suficient decrease, and the one in (71) is for the strong curvature, which rules out two things: short steps where the loss is still decreasing steeply, and long steps where the directional derivative becomes large and positive, see [32, Sec. 1, Eqs. (1.1)–(1.2)]. Setting $s _ { k } ~ = ~ \alpha _ { k } p _ { k }$ and $y _ { k } =$ $\nabla _ { \theta } \mathcal { L } ( \theta _ { k + 1 } ) - \nabla _ { \theta } \mathcal { L } ( \theta _ { k } )$ , the curvature condition implies $s _ { k } ^ { T } y _ { k } > 0$ . This keeps the BFGS inverse-Hessian approximation positive definite and preserves a descent direction in the next iteration, as shown in [33, Sec. 6.1, Eq. (6.8)]. Fixed steps or backtracking either depend more strongly on the learning rate or do not by themselves enforce $s _ { k } ^ { T } y _ { k } > 0$ , which is why the strong Wolfe search is used.

## References

[1] Agrawal, G. Nonlinear Fiber Optics, 3rd ed. Academic Press, San Diego, 2001.

[2] Bao, W., Chang, Z., and Zhao, X. Computing ground states of Bose–Einstein condensation by normalized deep neural network. J. Comput. Phys. 520 (2025), 113486.

[3] Baydin, A. G., Pearlmutter, B. A., Radul, A. A., and Siskind, J. M. Automatic diferentiation in machine learning: A survey. J. Mach. Learn. Res. 18, 153 (2018), 1–43.

[4] Berestycki, H., and Lions, P.-L. Nonlinear scalar field equations. I. existence of a ground state. Archive for Rational Mechanics and Analysis 82, 4 (1983), 313–345.

[5] Berrone, S., Canuto, C., Pintore, M., and Sukumar, N. Enforcing dirichlet boundary conditions in physics informed neural networks and variational physics-informed neural networks. Heliyon 9, 8 (2023), e18820.

[6] Blanco-Redondo, A., de Sterke, C. M., Xu, C., Wabnitz, S., and Turitsyn, S. K. The bright prospects of optical solitons after 50 years. Nature Photonics 17 (2023), 937–942.

[7] Bona, J. L., Souganidis, P. E., and Strauss, W. A. Stability and instability of solitary waves of Korteweg-de Vries type. Proc. Roy. Soc. London Ser. A 411, 1841 (1987), 395–412.

[8] Cazenave, T. Semilinear Schr¨odinger Equations, vol. 10 of Courant Lecture Notes in Mathematics. American Mathematical Society, Providence, RI, 2003.

[9] Chang, Z., Wen, Z., and Zhao, X. Deep neural network approaches for computing the defocusing action ground state of nonlinear Schr¨odinger equation. Ann. Appl. Math 41, 1 (2025), 42–76.

[10] E, W., and Yu, B. The Deep Ritz method: A deep learning-based numerical algorithm for solving variational problems. Comm. Math. Stat. 6, 1 (2018), 1–12.

[11] Gomes, A., and Pastor, A. Solitary wave solutions and global well-posedness for a coupled system of generalized Korteweg–de Vries equations. J. Evolut. Equ. 21 (2021), 2167–2193.

[12] Grillakis, M., Shatah, J., and Strauss, W. Stability theory of solitary waves in the presence of symmetry. I. Journal of Functional Analysis 74, 1 (1987), 160–197.

[13] Grillakis, M., Shatah, J., and Strauss, W. Stability theory of solitary waves in the presence of symmetry. II. Journal of Functional Analysis 94, 2 (1990), 308–348.

[14] Haight, C., Rodriguez, A. D., and Roudenko, S. Ground states in dispersive equations with combined nonlinearities, 2026. Preprint.

[15] Haight, C., Roudenko, S., and Wang, Z. Soliton profiles in 1D: Classical numerical schemes vs. neural network-based solvers. Chaos, Solitons and Fractals 210 (2026), 118646.

[16] Jnini, A., Vella, F., and Zeinhofer, M. Gauss-Newton natural gradient descent for physics-informed computational fluid dynamics, 2024. arXiv:2402.10680.

[17] Kenig, C. E., Ponce, G., and Vega, L. Well-posedness and scattering results for the generalized Korteweg–de Vries equation via the contraction principle. Comm. Pure Appl. Math. 46, 4 (1993), 527–620.

[18] Kingma, D. P., and Ba, J. Adam: A method for stochastic optimization. In International Conference on Learning Representations (2015). arXiv:1412.6980.

[19] Kivshar, Y., and Agrawal, G. Optical Solitons: From Fibers to Photonic Crystals. Academic Press, San Diego, 2003.

[20] Klein, C., and Roudenko, S. Ground states to biharmonic nonlinear Schr¨odinger equation with competing dispersion, 2026. arXiv:2609.13783.

[21] Klein, C., and Saut, J.-C. Nonlinear dispersive equations—inverse scattering and PDE methods, vol. 209 of Applied Mathematical Sciences. Springer, Cham, 2021.

[22] Kwong, M. K. Uniqueness of positive solutions of $\Delta u - u + u ^ { p } = 0$ in $\mathbf { R } ^ { n }$ . Arch. Rational Mech. Anal. 105, 3 (1989), 243–266.

[23] Lagaris, I. E., Likas, A., and Fotiadis, D. I. Artificial neural networks for solving ordinary and partial diferential equations. IEEE Transactions on Neural Networks 9, 5 (1998), 987–1000.

[24] Levenberg, K. A method for the solution of certain non-linear problems in least squares. Quarterly of Applied Mathematics 2, 2 (1944), 164–168.

[25] Li, X., Deng, J., Wu, J., Zhang, S., Li, W., and Wang, Y.-G. Physical informed neural networks with soft and hard boundary constraints for solving advection-difusion equations using Fourier expansions. Computers & Mathematics with Applications 159 (2024), 60–75.

[26] Linares, F., and Ponce, G. Introduction to Nonlinear Dispersive Equations, 2nd ed. Universitext. Springer, New York, 2015.

[27] Liu, D. C., and Nocedal, J. On the limited memory BFGS method for large scale optimization. Mathematical Programming 45, 1 (1989), 503–528.

[28] Liu, W., Wang, T., and Zhao, X. Convergence analysis of $L ^ { p + 1 }$ -normalized gradient flow for action ground state of nonlinear Schr¨odinger equation, 2026. arXiv:2602.20820.

[29] Liu, W., Yuan, Y., and Zhao, X. Computing the action ground state for the rotating nonlinear Schr¨odinger equation. SIAM Journal on Scientific Computing 45, 2 (2023), A397–A426.

[30] Lou, Q., Meng, X., and Karniadakis, G. E. Physics-informed neural networks for solving forward and inverse flow problems via the Boltzmann-BGK formulation. Journal of Computational Physics 447 (2021), 110676.

[31] Marquardt, D. W. An algorithm for least-squares estimation of nonlinear parameters. J. of SIAM 11, 2 (1963), 431–441.

[32] Mor’e, J. J., and Thuente, D. J. Line search algorithms with guaranteed suficient decrease. ACM Trans. Math. Software 20, 3 (1994), 286–307.

[33] Nocedal, J., and Wright, S. J. Numerical Optimization, 2 ed. Springer, New York, 2006.

[34] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N.<sub>,</sub> Antiga<sub>,</sub> L.<sub>,</sub> Desmaison<sub>,</sub> A.<sub>,</sub> Kopf<sub>,</sub> A.<sub>,</sub> Yang<sub>,</sub> E.<sub>,</sub> DeVito<sub>,</sub> Z.<sub>,</sub> Raison<sub>,</sub> M.<sub>,</sub> Tejani<sub>,</sub> A.<sub>,</sub> Chilamkurthy<sub>,</sub> S.<sub>,</sub>¨ Steiner, B., Fang, L., Bai, J., and Chintala, S. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems (2019), vol. 32, pp. 8024–8035.

[35] Rahimi, A., and Recht, B. Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems (2007), vol. 20.

[36] Raissi, M., Perdikaris, P., and Karniadakis, G. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. J. Comput. Phys. 378 (2019), 686–707.

[37] Rathore, P., Lei, W., Frangella, Z., Lu, L., and Udell, M. Challenges in training PINNs: A loss landscape perspective. In Proceedings of the 41st International Conference on Machine Learning (2024), vol. 235 of Proc. of Mach. Learn. Res., pp. 42159–42191.

[38] Sahin, T., von Danwitz, M., and Popp, A. Solving forward and inverse problems of contact mechanics using physics-informed neural networks. Advanced Modeling and Simulation in Engineering Sciences 11, 1 (2024), 11.

[39] Shahab, M. L., Mukhlash, I., and Susanto, H. Do physics-informed neural networks (PINNs) need to be deep? shallow PINNs using the Levenberg–Marquardt algorithm, 2026. arXiv:2602.08515.

[40] Soffer, A. Soliton dynamics and scattering. In International Congress of Mathematicians. Vol. III. Eur. Math. Soc., Z¨urich, 2006, pp. 459–471.

[41] Straub, C., Brendel, P., Medvedev, V., and Rosskopf, A. Hard-constraining neumann boundary condition in physics-informed neural networks via Fourier feature embeddings. In ICLR 2025 Workshop on Machine Learning Multiscale Processes (2025). arXiv:2504.01093.

[42] Sukumar, N., and Srivastava, A. Exact imposition of boundary conditions with distance functions in physics informed deep neural networks. Computer Methods in Applied Mechanics and Engineering 389 (2022), 114333.

[43] Sulem, C., and Sulem, P.-L. The Nonlinear Schr¨odinger Equation: Self-Focusing and Wave Collapse, vol. 139 of Applied Mathematical Sciences. Springer-Verlag, 1999.

[44] Tancik, M., Srinivasan, P. P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoor thi, R., Barron, J. T., and Ng, R. Fourier features let networks learn high frequency functions in low dimensional domains. In Advances in Neural Information Processing Systems (2020), vol. 33.

[45] Tao, T. Nonlinear Dispersive Equations: Local and Global Analysis, vol. 106 of CBMS Regional Conference Series in Mathematics. American Mathematical Society, Providence, RI, 2006.

[46] Tao, T. Why are solitons stable? Bull. Amer. Math. Soc. (N.S.) 46, 1 (2009), 1–33.

[47] Wang, C. Computing the least action ground state of the nonlinear Schr¨odinger equation by a normalized gradient flow. J. Comput. Phys. 471 (2022), 111675.

[48] Wang, S., Wang, H., and Perdikaris, P. On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale pdes with physics-informed neural networks. Comput. Meth. Appl. Mech. Eng. 384 (2021), 113938.

[49] Wang, Y., Bennani, M., Martens, J., Racaniere, S., Blackwell, S., Matthews, A., Nikolov, S., Cao-<sup>\`</sup> Labora, G., Park, D. S., Arjovsky, M., Worrall, D., Qin, C., Alet, F., Kozlovskii, B., Tomasev, N., Davies, A., Kohli, P., Buckmaster, T., Georgiev, B., Gomez-Serrano, J., Jiang, R., and Lai, C.-Y.<sup>´</sup> Discovery of unstable singularities, 2025. arXiv:2509.14185.

[50] Weinstein, M. I. Nonlinear schr¨odinger equations and sharp interpolation estimates. Comm. Math. Phys. 87, 4 (1983), 567–576.

[51] Weinstein, M. I. Modulational stability of ground states of nonlinear Schr¨odinger equations. SIAM Journal on Mathematical Analysis 16, 3 (1985), 472–491.

[52] Weinstein, M. I. Lyapunov stability of ground states of nonlinear dispersive evolution equations. Comm. Pure Appl. Math. 39, 1 (1986), 51–67.

[53] Wolfe, P. Convergence conditions for ascent methods. SIAM Review 11, 2 (1969), 226–235.

[54] Wolfe, P. Convergence conditions for ascent methods. II: Some corrections. SIAM Review 13, 2 (1971), 185–188.

[55] Yao, Y., Lu, Y., and Lerman, G. Feature learning for the high dimensional stationary Schr¨odinger equation with Deep Ritz method, 2026. arXiv:2607.06369.

[56] Zakharov, V. Collapse of Langmuir waves. Zh. Eksp. Teor. Fiz. 62 (1972), 1745–1751 [transl. in Sov. Phys. JETP 35, 908–914 (1972)].

[57] Zakharov, V., and Rubenchik, A. Instability of waveguides and solitons in nonlinear media. Zh. Eksp. Teor. Fiz. 65 (1973), 997–1011 [transl. in Sov. Phys. JETP 38, 1974, no. 3, 494–500].

Department of Mathematics & Statistics<sub>,</sub> Florida International University<sub>,</sub> Miami<sub>,</sub> FL 33199<sub>,</sub> USA Email address: chaig004@fiu.edu

Department of Mathematics & Statistics<sub>,</sub> Florida International University<sub>,</sub> Miami<sub>,</sub> FL 33199<sub>,</sub> USA Email address: arodr1128@fiu.edu

Department of Mathematics & Statistics<sub>,</sub> Florida International University<sub>,</sub> Miami<sub>,</sub> FL 33199<sub>,</sub> USA Email address: sroudenko@fiu.edu